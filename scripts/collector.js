import { readFileSync, unlinkSync, existsSync, appendFileSync, writeFileSync, } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadPluginConfig } from "./config.js";
import { serializeAttribute } from "./serialize.js";
// @sentry/node is loaded lazily: importing it costs well over a second, and the
// vast majority of hook invocations (PreToolUse/PostToolUse/UserPromptSubmit in
// batch mode) only append a line to a JSONL file and never talk to Sentry.
// Only the paths that actually emit spans call initSentry().
let Sentry;
async function initSentry(config) {
    Sentry = await import("@sentry/node");
    Sentry.init({
        dsn: config.dsn,
        tracesSampleRate: config.tracesSampleRate,
        environment: config.environment,
        release: config.release,
        debug: config.debug,
    });
    if (config.user) {
        Sentry.setUser(config.user);
    }
}
// ── Helpers ──────────────────────────────────────────────────
function safeJsonParse(str) {
    try {
        return JSON.parse(str);
    }
    catch {
        return null;
    }
}
function addTimestamp(event) {
    return { ...event, _ts: Date.now() };
}
/**
 * Per-session batch log path. Uses the OS temp dir so it works on Windows too
 * (the old hardcoded "/tmp" silently dropped everything on Windows).
 */
function batchLogPath(sessionId) {
    return join(tmpdir(), `claude-sentry-${sessionId}.jsonl`);
}
/**
 * Per-session token offsets. Deliberately kept OUTSIDE the batch logfile, which
 * is deleted at SessionEnd: the transcript accumulates across resumes, so a
 * session that is resumed would otherwise re-report its entire token history
 * every time it is reopened.
 */
function offsetPath(sessionId) {
    return join(tmpdir(), `claude-sentry-${sessionId}.offset.json`);
}
function extractTokensFromTranscript(transcriptPath) {
    if (!existsSync(transcriptPath))
        return null;
    let inputTokens = 0;
    let cacheCreationTokens = 0;
    let cacheReadTokens = 0;
    let outputTokens = 0;
    let model = null;
    let prompt = null;
    let lastResponse = null;
    const turnResponses = [];
    let currentTurnResponse = null;
    let inTurn = false;
    const content = readFileSync(transcriptPath, "utf-8");
    for (const line of content.split("\n")) {
        if (!line)
            continue;
        const obj = safeJsonParse(line);
        if (!obj)
            continue;
        // Capture first user message as prompt; track turn boundaries
        if (obj.type === "user") {
            const msg = obj.message?.content ?? obj.message;
            if (!prompt) {
                prompt = typeof msg === "string" ? msg : JSON.stringify(msg);
            }
            // Close previous turn
            if (inTurn && currentTurnResponse !== null) {
                turnResponses.push(currentTurnResponse);
            }
            currentTurnResponse = null;
            inTurn = true;
        }
        // Capture assistant text responses
        if (obj.type === "assistant" && Array.isArray(obj.message?.content)) {
            const texts = obj.message.content
                .filter((c) => c.type === "text" && c.text)
                .map((c) => c.text);
            if (texts.length) {
                const response = texts.join("\n");
                lastResponse = response;
                if (inTurn) {
                    currentTurnResponse = response; // keep updating to get the last one
                }
            }
        }
        if (obj.type !== "assistant" || !obj.message?.usage)
            continue;
        const usage = obj.message.usage;
        // Keep the three input buckets separate: they have very different costs, and
        // in long sessions cache reads dwarf everything else (the same context is
        // re-read every turn). Collapsing them into one number makes Sentry price
        // cached tokens at the full input rate.
        inputTokens += usage.input_tokens || 0;
        cacheCreationTokens += usage.cache_creation_input_tokens || 0;
        cacheReadTokens += usage.cache_read_input_tokens || 0;
        outputTokens += usage.output_tokens || 0;
        if (obj.message.model) {
            model = obj.message.model;
        }
    }
    // Close last turn
    if (inTurn && currentTurnResponse !== null) {
        turnResponses.push(currentTurnResponse);
    }
    return {
        inputTokens,
        cacheCreationTokens,
        cacheReadTokens,
        outputTokens,
        model,
        prompt,
        lastResponse,
        turnResponses,
    };
}
function pairToolEvents(events) {
    const preByUseId = new Map();
    const preByToolName = new Map();
    const completed = [];
    for (const event of events) {
        if (event.hook_event_name === "PreToolUse") {
            if (event.tool_use_id) {
                preByUseId.set(event.tool_use_id, event);
            }
            else {
                const stack = preByToolName.get(event.tool_name) || [];
                stack.push(event);
                preByToolName.set(event.tool_name, stack);
            }
        }
        else if (event.hook_event_name === "PostToolUse") {
            let pre;
            if (event.tool_use_id) {
                pre = preByUseId.get(event.tool_use_id);
                if (pre)
                    preByUseId.delete(event.tool_use_id);
            }
            else {
                const stack = preByToolName.get(event.tool_name);
                if (stack?.length)
                    pre = stack.pop();
            }
            const startTime = pre ? pre._ts : event._ts - 1;
            completed.push({
                tool_name: event.tool_name,
                startTime,
                endTime: event._ts,
                input: pre?.tool_input ?? event.tool_input,
                output: event.tool_response,
                tool_error: event.tool_error === true,
            });
        }
    }
    return completed;
}
// ── Tool span creation helper ────────────────────────────────
function createToolSpan(tool, config) {
    const attrs = {
        "gen_ai.tool.name": tool.tool_name,
    };
    if (config.recordInputs && tool.input) {
        attrs["gen_ai.tool.input"] = serializeAttribute(tool.input, config.maxAttributeLength);
    }
    if (config.recordOutputs && tool.output) {
        attrs["gen_ai.tool.output"] = serializeAttribute(tool.output, config.maxAttributeLength);
    }
    if (tool.tool_error) {
        attrs["gen_ai.tool.error"] = true;
    }
    const childSpan = Sentry.startInactiveSpan({
        name: `execute_tool ${tool.tool_name}`,
        op: "gen_ai.execute_tool",
        startTime: tool.startTime,
        attributes: attrs,
    });
    if (tool.tool_error) {
        childSpan.setStatus({ code: 2, message: "tool_error" });
    }
    childSpan.end(tool.endTime);
}
const ZERO_CARRY = {
    chapterIndex: 0,
    offset: { input: 0, cacheWrite: 0, cacheRead: 0, output: 0 },
};
/**
 * Read the chapter index and token totals already reported for this session.
 * Survives SessionEnd so that resuming a session reports only new tokens.
 */
function readCarry(sessionId) {
    try {
        const raw = readFileSync(offsetPath(sessionId), "utf-8");
        const parsed = safeJsonParse(raw);
        if (!parsed)
            return ZERO_CARRY;
        return {
            chapterIndex: parsed.chapter || 0,
            offset: {
                input: parsed.input || 0,
                cacheWrite: parsed.cacheWrite || 0,
                cacheRead: parsed.cacheRead || 0,
                output: parsed.output || 0,
            },
        };
    }
    catch {
        return ZERO_CARRY;
    }
}
function writeCarry(sessionId, chapterIndex, totals) {
    try {
        writeFileSync(offsetPath(sessionId), JSON.stringify({ chapter: chapterIndex, ...totals }));
    }
    catch { }
}
/**
 * Build the transaction tree for a set of events and flush it to Sentry.
 * Returns the cumulative transcript token totals observed, so the caller can
 * carry them forward as the next chapter's offset (avoids double-counting
 * tokens, since the transcript accumulates across chapters).
 */
async function emitTransaction(events, config, carry, ongoing) {
    const sessionStart = events.find((e) => e.hook_event_name === "SessionStart");
    const model = sessionStart?.model || events[0]?.model || "claude";
    const sessionId = sessionStart?.session_id || events[0]?.session_id;
    const transcriptPath = sessionStart?.transcript_path || events[0]?.transcript_path;
    const tokenData = transcriptPath ? extractTokensFromTranscript(transcriptPath) : null;
    const cumulative = {
        input: tokenData?.inputTokens ?? carry.offset.input,
        cacheWrite: tokenData?.cacheCreationTokens ?? carry.offset.cacheWrite,
        cacheRead: tokenData?.cacheReadTokens ?? carry.offset.cacheRead,
        output: tokenData?.outputTokens ?? carry.offset.output,
    };
    const toolCalls = pairToolEvents(events);
    const firstTs = events[0]._ts || Date.now() / 1000;
    const lastTs = events[events.length - 1]._ts || Date.now() / 1000;
    // Find UserPromptSubmit event indices for turn-based tracing
    const userPromptIndices = events
        .map((e, i) => (e.hook_event_name === "UserPromptSubmit" ? i : -1))
        .filter((i) => i >= 0);
    const rootAttrs = {
        "gen_ai.agent.name": "claude-code",
        "gen_ai.request.model": model,
        "gen_ai.system": "anthropic",
    };
    // Add custom tags
    for (const [key, value] of Object.entries(config.tags)) {
        rootAttrs[key] = value;
    }
    // Chapter metadata (only meaningful when chunking is enabled). session_id lets
    // you group every chapter of one long session in Sentry.
    if (sessionId) {
        rootAttrs["claude.session_id"] = sessionId;
    }
    if (carry.chapterIndex > 0 || ongoing) {
        rootAttrs["claude.session_chapter"] = carry.chapterIndex;
        rootAttrs["claude.session_ongoing"] = ongoing;
    }
    const rootSpan = Sentry.startInactiveSpan({
        name: "invoke_agent claude-code",
        op: "gen_ai.invoke_agent",
        forceTransaction: true,
        startTime: firstTs,
        attributes: rootAttrs,
    });
    // Set token data from session transcript, reported as this chapter's delta.
    if (tokenData) {
        const freshInput = Math.max(0, cumulative.input - carry.offset.input);
        const cacheWrite = Math.max(0, cumulative.cacheWrite - carry.offset.cacheWrite);
        const cacheRead = Math.max(0, cumulative.cacheRead - carry.offset.cacheRead);
        const chapterOutput = Math.max(0, cumulative.output - carry.offset.output);
        // Sentry's cost formula is:
        //   (input_tokens - cached) * input_rate
        //   + cached * cached_rate
        //   + cache_write * cache_write_rate
        // so input_tokens must include cached reads but NOT cache writes, otherwise
        // cache writes get billed twice. (OTel semconv would fold cache_creation in
        // as well; that double-counts under Sentry's formula.)
        const totalInput = freshInput + cacheRead;
        if (totalInput) {
            rootSpan.setAttribute("gen_ai.usage.input_tokens", totalInput);
        }
        if (cacheRead) {
            rootSpan.setAttribute("gen_ai.usage.input_tokens.cached", cacheRead);
        }
        if (cacheWrite) {
            rootSpan.setAttribute("gen_ai.usage.input_tokens.cache_write", cacheWrite);
        }
        if (chapterOutput) {
            rootSpan.setAttribute("gen_ai.usage.output_tokens", chapterOutput);
        }
        if (totalInput || chapterOutput) {
            rootSpan.setAttribute("gen_ai.usage.total_tokens", totalInput + chapterOutput);
        }
        if (tokenData.model) {
            rootSpan.setAttribute("gen_ai.response.model", tokenData.model);
        }
        // Only set flat input/output on root span when there are no per-turn spans
        if (userPromptIndices.length === 0) {
            if (config.recordInputs && tokenData.prompt) {
                rootSpan.setAttribute("gen_ai.request.messages", serializeAttribute([{ role: "user", content: tokenData.prompt }], config.maxAttributeLength));
            }
            if (config.recordOutputs && tokenData.lastResponse) {
                rootSpan.setAttribute("gen_ai.response.text", serializeAttribute([tokenData.lastResponse], config.maxAttributeLength));
            }
        }
    }
    if (userPromptIndices.length > 0) {
        // Turn-based mode: create gen_ai.chat span per conversation turn
        Sentry.withActiveSpan(rootSpan, () => {
            for (let t = 0; t < userPromptIndices.length; t++) {
                const startIdx = userPromptIndices[t];
                const endIdx = t + 1 < userPromptIndices.length ? userPromptIndices[t + 1] : events.length;
                const turnEvents = events.slice(startIdx, endIdx);
                const promptEvent = events[startIdx];
                const turnStartTs = promptEvent._ts;
                const turnEndTs = t + 1 < userPromptIndices.length
                    ? events[userPromptIndices[t + 1]]._ts
                    : lastTs;
                const turnPrompt = promptEvent.prompt || promptEvent.message || null;
                const turnResponse = tokenData?.turnResponses[t] ?? null;
                const turnAttrs = {};
                if (config.recordInputs && turnPrompt) {
                    turnAttrs["gen_ai.request.messages"] = serializeAttribute([{ role: "user", content: turnPrompt }], config.maxAttributeLength);
                }
                if (config.recordOutputs && turnResponse) {
                    turnAttrs["gen_ai.response.text"] = serializeAttribute([turnResponse], config.maxAttributeLength);
                }
                const turnSpan = Sentry.startInactiveSpan({
                    name: "gen_ai.chat",
                    op: "gen_ai.request",
                    startTime: turnStartTs,
                    attributes: turnAttrs,
                });
                const turnToolCalls = pairToolEvents(turnEvents);
                Sentry.withActiveSpan(turnSpan, () => {
                    for (const tool of turnToolCalls) {
                        createToolSpan(tool, config);
                    }
                });
                turnSpan.end(turnEndTs);
            }
        });
    }
    else {
        // Fallback: flat tool spans directly under root span
        Sentry.withActiveSpan(rootSpan, () => {
            for (const tool of toolCalls) {
                createToolSpan(tool, config);
            }
        });
    }
    rootSpan.setAttribute("gen_ai.tool.call_count", toolCalls.length);
    rootSpan.end(lastTs);
    await Sentry.flush(10_000);
    return cumulative;
}
/** Read events from a batch logfile, dropping blank/corrupt lines. */
function readBatchEvents(filePath) {
    if (!existsSync(filePath))
        return [];
    const lines = readFileSync(filePath, "utf-8").trim().split("\n");
    return lines.map((line) => safeJsonParse(line)).filter(Boolean);
}
/**
 * Final flush (SessionEnd): emit remaining events, persist the token totals so a
 * later resume of this session doesn't re-report them, then remove the logfile.
 */
async function processBatch(filePath, config, sessionId) {
    const events = readBatchEvents(filePath);
    if (events.length === 0) {
        try {
            unlinkSync(filePath);
        }
        catch { }
        return;
    }
    const carry = readCarry(sessionId);
    const cumulative = await emitTransaction(events, config, carry, false);
    writeCarry(sessionId, carry.chapterIndex + 1, cumulative);
    try {
        unlinkSync(filePath);
    }
    catch { }
}
/**
 * Time-triggered flush for a still-open session: emit everything so far as a
 * "chapter" transaction, then reset the logfile to a single synthetic
 * SessionStart that carries the model, transcript path and token offset forward
 * so the next chapter is a fresh, correctly-accounted transaction.
 */
async function processChunk(filePath, config, sessionId) {
    const events = readBatchEvents(filePath);
    // Nothing worth a chapter unless there's real activity beyond the marker.
    if (events.length <= 1)
        return;
    const carry = readCarry(sessionId);
    const cumulative = await emitTransaction(events, config, carry, true);
    writeCarry(sessionId, carry.chapterIndex + 1, cumulative);
    const start = events.find((e) => e.hook_event_name === "SessionStart");
    const carried = {
        hook_event_name: "SessionStart",
        session_id: sessionId,
        model: start?.model || events[0]?.model,
        transcript_path: start?.transcript_path || events[0]?.transcript_path,
        _ts: Date.now(),
    };
    try {
        writeFileSync(filePath, JSON.stringify(carried) + "\n");
    }
    catch { }
}
/** Age in ms of the current chapter (its first/SessionStart event). */
function chapterAgeMs(filePath) {
    const events = readBatchEvents(filePath);
    if (events.length === 0)
        return 0;
    const firstTs = events[0]._ts || Date.now();
    return Date.now() - firstTs;
}
// ── Real-time server mode ────────────────────────────────────
function startServer(config) {
    const PORT = parseInt(process.env.SENTRY_COLLECTOR_PORT || "9876", 10);
    const sessions = new Map();
    function newRootSpan(sessionId, model, chapter, ongoing) {
        const rootAttrs = {
            "gen_ai.agent.name": "claude-code",
            "gen_ai.request.model": model,
            "gen_ai.system": "anthropic",
            "claude.session_id": sessionId,
        };
        for (const [key, value] of Object.entries(config.tags)) {
            rootAttrs[key] = value;
        }
        if (chapter > 0 || ongoing) {
            rootAttrs["claude.session_chapter"] = chapter;
            rootAttrs["claude.session_ongoing"] = ongoing;
        }
        return Sentry.startInactiveSpan({
            name: "invoke_agent claude-code",
            op: "gen_ai.invoke_agent",
            forceTransaction: true,
            attributes: rootAttrs,
        });
    }
    /**
     * Close the current chapter of a still-open session and start the next one,
     * so realtime sessions that never end still report every flushInterval.
     */
    function rotateSession(sessionId, session) {
        if (session.currentTurnSpan) {
            session.currentTurnSpan.end();
            session.currentTurnSpan = null;
        }
        for (const span of session.pendingTools.values()) {
            span.end();
        }
        session.pendingTools.clear();
        session.rootSpan.setAttribute("gen_ai.tool.call_count", session.toolCount);
        session.rootSpan.setAttribute("claude.session_ongoing", true);
        session.rootSpan.end();
        Sentry.flush(5_000);
        session.chapter += 1;
        session.toolCount = 0;
        session.rootSpan = newRootSpan(sessionId, session.model, session.chapter, true);
    }
    function handleEvent(event) {
        const { session_id, hook_event_name: rawEvent, tool_name } = event;
        const hook_event_name = rawEvent.charAt(0).toUpperCase() + rawEvent.slice(1);
        switch (hook_event_name) {
            case "SessionStart": {
                const model = event.model || "claude";
                sessions.set(session_id, {
                    rootSpan: newRootSpan(session_id, model, 0, false),
                    currentTurnSpan: null,
                    pendingTools: new Map(),
                    toolCount: 0,
                    model,
                    chapter: 0,
                });
                break;
            }
            case "UserPromptSubmit": {
                const session = sessions.get(session_id);
                if (!session)
                    break;
                // End previous turn span if any
                if (session.currentTurnSpan) {
                    session.currentTurnSpan.end();
                }
                const turnAttrs = {};
                const prompt = event.prompt || event.message || null;
                if (config.recordInputs && prompt) {
                    turnAttrs["gen_ai.request.messages"] = serializeAttribute([{ role: "user", content: prompt }], config.maxAttributeLength);
                }
                session.currentTurnSpan = Sentry.withActiveSpan(session.rootSpan, () => Sentry.startInactiveSpan({
                    name: "gen_ai.chat",
                    op: "gen_ai.request",
                    attributes: turnAttrs,
                }));
                break;
            }
            case "PreToolUse": {
                const session = sessions.get(session_id);
                if (!session)
                    break;
                const attrs = {
                    "gen_ai.tool.name": tool_name ?? "unknown",
                };
                if (config.recordInputs && event.tool_input) {
                    attrs["gen_ai.tool.input"] = serializeAttribute(event.tool_input, config.maxAttributeLength);
                }
                const parentSpan = session.currentTurnSpan ?? session.rootSpan;
                const toolSpan = Sentry.withActiveSpan(parentSpan, () => Sentry.startInactiveSpan({
                    name: `execute_tool ${tool_name}`,
                    op: "gen_ai.execute_tool",
                    attributes: attrs,
                }));
                if (event.tool_use_id) {
                    session.pendingTools.set(event.tool_use_id, toolSpan);
                }
                session.toolCount++;
                break;
            }
            case "PostToolUse": {
                const session = sessions.get(session_id);
                if (!session)
                    break;
                const toolSpan = event.tool_use_id
                    ? session.pendingTools.get(event.tool_use_id)
                    : undefined;
                if (toolSpan) {
                    if (config.recordOutputs && event.tool_response) {
                        toolSpan.setAttribute("gen_ai.tool.output", serializeAttribute(event.tool_response, config.maxAttributeLength));
                    }
                    if (event.tool_error === true) {
                        toolSpan.setAttribute("gen_ai.tool.error", true);
                        toolSpan.setStatus({ code: 2, message: "tool_error" });
                    }
                    toolSpan.end();
                    session.pendingTools.delete(event.tool_use_id);
                }
                break;
            }
            case "SessionEnd": {
                const session = sessions.get(session_id);
                if (!session)
                    break;
                if (session.currentTurnSpan) {
                    session.currentTurnSpan.end();
                }
                for (const span of session.pendingTools.values()) {
                    span.end();
                }
                session.rootSpan.setAttribute("gen_ai.tool.call_count", session.toolCount);
                session.rootSpan.end();
                sessions.delete(session_id);
                Sentry.flush(5_000);
                break;
            }
        }
    }
    const server = createServer((req, res) => {
        if (req.url === "/health") {
            res.writeHead(200);
            res.end("ok");
            return;
        }
        if (req.url !== "/hook" || req.method !== "POST") {
            res.writeHead(404);
            res.end("not found");
            return;
        }
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
            try {
                handleEvent(JSON.parse(body));
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end("{}");
            }
            catch (err) {
                res.writeHead(400);
                res.end(err.message);
            }
        });
    });
    server.listen(PORT, "127.0.0.1", () => {
        // silent
    });
    // Periodically rotate open sessions so ones that never end still report.
    if (config.flushIntervalMinutes > 0) {
        const timer = setInterval(() => {
            for (const [sessionId, session] of sessions) {
                rotateSession(sessionId, session);
            }
        }, config.flushIntervalMinutes * 60_000);
        timer.unref();
    }
    process.on("SIGTERM", async () => {
        server.close();
        for (const [, session] of sessions) {
            session.rootSpan.end();
        }
        await Sentry.flush(5_000);
        process.exit(0);
    });
}
// ── Main entry point (reads stdin) ───────────────────────────
async function main() {
    // Read hook event from stdin
    const chunks = [];
    for await (const chunk of process.stdin) {
        chunks.push(chunk);
    }
    const inputStr = Buffer.concat(chunks).toString("utf-8").trim();
    if (!inputStr) {
        process.exit(0);
    }
    const event = safeJsonParse(inputStr);
    if (!event) {
        process.exit(0);
    }
    // Load config
    const loaded = await loadPluginConfig();
    if (!loaded) {
        // No DSN configured, exit silently
        process.exit(0);
    }
    const { config } = loaded;
    const timestamped = addTimestamp(event);
    // Normalize: Claude Code sends "sessionEnd" (camelCase) but other events are PascalCase
    const rawHookEvent = event.hook_event_name;
    const hookEvent = rawHookEvent.charAt(0).toUpperCase() + rawHookEvent.slice(1);
    const sessionId = event.session_id;
    if (!sessionId) {
        process.exit(0);
    }
    if (config.mode === "realtime") {
        // In realtime mode, forward to collector server
        const PORT = parseInt(process.env.SENTRY_COLLECTOR_PORT || "9876", 10);
        const BASE = `http://127.0.0.1:${PORT}`;
        if (hookEvent === "SessionStart") {
            // Ensure collector server is running
            try {
                const healthRes = await fetch(`${BASE}/health`);
                if (!healthRes.ok)
                    throw new Error("not ok");
            }
            catch {
                // Start server in background
                const { spawn } = await import("node:child_process");
                const child = spawn(process.execPath, [import.meta.filename, "--serve", JSON.stringify(config)], {
                    detached: true,
                    stdio: "ignore",
                });
                child.unref();
                // Wait for server to be ready
                for (let i = 0; i < 6; i++) {
                    await new Promise((r) => setTimeout(r, 500));
                    try {
                        const res = await fetch(`${BASE}/health`);
                        if (res.ok)
                            break;
                    }
                    catch { }
                }
            }
        }
        // POST event to collector
        try {
            await fetch(`${BASE}/hook`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(timestamped),
            });
        }
        catch { }
    }
    else {
        // Batch mode: append to session-specific JSONL file
        const logfile = batchLogPath(sessionId);
        appendFileSync(logfile, JSON.stringify(timestamped) + "\n");
        if (hookEvent === "SessionEnd") {
            // Final flush of whatever remains in the current chapter.
            await initSentry(config);
            await processBatch(logfile, config, sessionId);
        }
        else if (config.flushIntervalMinutes > 0 &&
            chapterAgeMs(logfile) >= config.flushIntervalMinutes * 60_000) {
            // Long-lived session: flush the current chapter and start a fresh one so
            // sessions that never end still report periodically.
            await initSentry(config);
            await processChunk(logfile, config, sessionId);
        }
        // Otherwise we only appended a line — Sentry was never loaded.
    }
}
// Handle --serve flag (spawned by realtime mode)
const [, , command, configArg] = process.argv;
if (command === "--serve" && configArg) {
    const config = JSON.parse(configArg);
    initSentry(config)
        .then(() => startServer(config))
        .catch(() => process.exit(0));
}
else {
    main().catch(() => process.exit(0));
}
