package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func interactionsUserText(text string) schemas.ResponsesMessage {
	return userResponsesMessage(&schemas.ResponsesMessageContent{ContentStr: schemas.Ptr(text)})
}

func marshalInteractionsBody(t *testing.T, req *GeminiInteractionsRequest) string {
	t.Helper()
	raw, err := providerUtils.MarshalSorted(req)
	require.NoError(t, err)
	return string(raw)
}

// TestShouldUseGeminiInteractions pins the gate: only the Interactions route, a stored previous
// turn, a background run, or an agent leave generateContent.
func TestShouldUseGeminiInteractions(t *testing.T) {
	plain := &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	assert.False(t, ShouldUseGeminiInteractions(ctx, plain))
	assert.False(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("")}}))

	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("v1_abc")}}))
	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true)}}))
	assert.True(t, ShouldUseGeminiInteractions(ctx, &schemas.BifrostResponsesRequest{Model: "deep-research-preview-04-2026"}))

	routed := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	routed.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	assert.True(t, ShouldUseGeminiInteractions(routed, plain))
}

func TestToGeminiInteractionsRequestSingleTurn(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleDeveloper), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Be brief.")}},
			interactionsUserText("What is the capital of France?"),
		},
		Params: &schemas.ResponsesParameters{
			Instructions:       schemas.Ptr("You are a geographer."),
			PreviousResponseID: schemas.Ptr("v1_prev"),
			Store:              schemas.Ptr(true),
			Background:         schemas.Ptr(false),
			MaxOutputTokens:    schemas.Ptr(256),
			Temperature:        schemas.Ptr(0.2),
			Reasoning:          &schemas.ResponsesParametersReasoning{Effort: schemas.Ptr("high"), Summary: schemas.Ptr("auto")},
			ServiceTier:        schemas.Ptr(schemas.BifrostServiceTierFlex),
			ToolChoice:         &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &schemas.ResponsesToolChoiceStruct{Type: schemas.ResponsesToolChoiceTypeFunction, Name: schemas.Ptr("get_weather")}},
			Tools: []schemas.ResponsesTool{
				{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather"), Description: schemas.Ptr("Weather by city"), ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &schemas.ToolFunctionParameters{Type: "object", Required: []string{"city"}}}},
				{Type: schemas.ResponsesToolTypeWebSearch},
			},
			ExtraParams: map[string]interface{}{"seed": 7, "stop": []string{"END"}},
		},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"model":"gemini-3.8-flash",
		"input":"What is the capital of France?",
		"system_instruction":"You are a geographer.\n\nBe brief.",
		"previous_interaction_id":"v1_prev",
		"store":true,
		"background":false,
		"service_tier":"flex",
		"tools":[
			{"type":"function","name":"get_weather","description":"Weather by city","parameters":{"type":"object","properties":{},"required":["city"]}},
			{"type":"google_search"}
		],
		"generation_config":{
			"max_output_tokens":256,"temperature":0.2,"seed":7,"stop_sequences":["END"],
			"thinking_level":"high","thinking_summaries":"auto",
			"tool_choice":{"allowed_tools":{"mode":"any","tools":["get_weather"]}}
		}
	}`, marshalInteractionsBody(t, native))
	assert.Nil(t, native.ExtraParams, "seed and stop are moved into generation_config")
}

// TestToGeminiInteractionsRequestHistoryAsSteps pins stateless replay: a history with model turns,
// reasoning and tool traffic is sent as steps, with the thought signature kept.
func TestToGeminiInteractionsRequestHistoryAsSteps(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			interactionsUserText("Weather in Paris?"),
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning), ResponsesReasoning: &schemas.ResponsesReasoning{EncryptedContent: schemas.Ptr("sig_1"), Summary: []schemas.ResponsesReasoningSummary{{Type: schemas.ResponsesReasoningContentBlockTypeSummaryText, Text: "Need weather."}}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Arguments: schemas.Ptr(`{"city":"Paris"}`)}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr(`{"temp":21}`)}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant), Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("It is 21C.")}}}},
			interactionsUserText("Thanks"),
		},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"model":"gemini-3.8-flash",
		"input":[
			{"type":"user_input","content":[{"type":"text","text":"Weather in Paris?"}]},
			{"type":"thought","signature":"sig_1","summary":[{"type":"text","text":"Need weather."}]},
			{"type":"function_call","id":"call_1","name":"get_weather","arguments":{"city":"Paris"}},
			{"type":"function_result","call_id":"call_1","name":"get_weather","result":[{"type":"text","text":"{\"temp\":21}"}]},
			{"type":"model_output","content":[{"type":"text","text":"It is 21C."}]},
			{"type":"user_input","content":[{"type":"text","text":"Thanks"}]}
		]
	}`, marshalInteractionsBody(t, native))
}

func TestToGeminiInteractionsRequestAgentAndMedia(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model: "deep-research-preview-04-2026",
		Input: []schemas.ResponsesMessage{userResponsesMessage(&schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{
			{Type: schemas.ResponsesInputMessageContentBlockTypeText, Text: schemas.Ptr("Summarize")},
			{Type: schemas.ResponsesInputMessageContentBlockTypeImage, ResponsesInputMessageContentBlockImage: &schemas.ResponsesInputMessageContentBlockImage{ImageURL: schemas.Ptr("data:image/png;base64,AAAA")}},
			{Type: schemas.ResponsesInputMessageContentBlockTypeFile, ResponsesInputMessageContentBlockFile: &schemas.ResponsesInputMessageContentBlockFile{FileURL: schemas.Ptr("https://example.com/report.pdf")}},
		}})},
		Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true), Text: &schemas.ResponsesTextConfig{Format: &schemas.ResponsesTextConfigFormat{Type: "json_object"}}},
	}
	native, err := ToGeminiInteractionsRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"agent":"deep-research-preview-04-2026",
		"background":true,
		"input":[
			{"type":"text","text":"Summarize"},
			{"type":"image","data":"AAAA","mime_type":"image/png"},
			{"type":"document","uri":"https://example.com/report.pdf","mime_type":"application/pdf"}
		],
		"response_format":{"type":"text","mime_type":"application/json"}
	}`, marshalInteractionsBody(t, native))
}

func TestToGeminiInteractionsRequestRefusesUnservedTool(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Model:  "gemini-3.8-flash",
		Input:  []schemas.ResponsesMessage{interactionsUserText("hi")},
		Params: &schemas.ResponsesParameters{Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeLocalShell}}},
	}
	_, err := ToGeminiInteractionsRequest(req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"local_shell"`)
}

func TestGeminiInteractionsInputUnmarshal(t *testing.T) {
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":"hello"}`), &req))
	require.NotNil(t, req.Input.Text)
	assert.Equal(t, "hello", *req.Input.Text)

	req = GeminiInteractionsRequest{}
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":{"type":"function_result","call_id":"c1","result":"ok"}}`), &req))
	require.Len(t, req.Input.Items, 1)
	assert.Equal(t, "function_result", req.Input.Items[0].Type)

	req = GeminiInteractionsRequest{}
	require.NoError(t, sonic.Unmarshal([]byte(`{"model":"m","input":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"stream":true}`), &req))
	assert.Len(t, req.Input.Items, 2)
	assert.True(t, req.IsStreamingRequested())
}

// TestGeminiInteractionsRequestToBifrost pins the inbound translation used when an Interactions
// caller's request runs on another provider.
func TestGeminiInteractionsRequestToBifrost(t *testing.T) {
	body := `{
		"model":"openai/gpt-5",
		"system_instruction":"Be brief.",
		"previous_interaction_id":"resp_1",
		"input":[
			{"type":"function_result","call_id":"call_1","name":"get_weather","result":[{"type":"text","text":"21C"}]},
			{"type":"text","text":"and tomorrow?"}
		],
		"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],
		"generation_config":{"max_output_tokens":100,"thinking_level":"low","tool_choice":"any","seed":3},
		"response_format":{"type":"text","mime_type":"application/json","schema":{"type":"object"}},
		"service_tier":"priority",
		"fallbacks":["anthropic/claude-sonnet-5-5"]
	}`
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(body), &req))
	require.NoError(t, req.ValidateTranslatable())
	out, err := req.ToBifrostResponsesRequest()
	require.NoError(t, err)
	assert.Equal(t, schemas.OpenAI, out.Provider)
	assert.Equal(t, "gpt-5", out.Model)
	require.Len(t, out.Fallbacks, 1)
	assert.Equal(t, schemas.Anthropic, out.Fallbacks[0].Provider)

	p := out.Params
	assert.Equal(t, "Be brief.", *p.Instructions)
	assert.Equal(t, "resp_1", *p.PreviousResponseID)
	assert.Equal(t, 100, *p.MaxOutputTokens)
	assert.Equal(t, "low", *p.Reasoning.Effort)
	assert.Equal(t, "required", *p.ToolChoice.ResponsesToolChoiceStr)
	assert.Equal(t, 3, p.ExtraParams["seed"])
	assert.Equal(t, schemas.BifrostServiceTierPriority, *p.ServiceTier)
	assert.Equal(t, "json_schema", p.Text.Format.Type)
	require.Len(t, p.Tools, 1)
	assert.Equal(t, "get_weather", *p.Tools[0].Name)

	require.Len(t, out.Input, 2)
	assert.Equal(t, schemas.ResponsesMessageTypeFunctionCallOutput, *out.Input[0].Type)
	assert.Equal(t, "call_1", *out.Input[0].CallID)
	assert.Equal(t, "21C", *out.Input[0].Output.ResponsesToolCallOutputStr)
	assert.Equal(t, schemas.ResponsesInputMessageRoleUser, *out.Input[1].Role)
	assert.Equal(t, "and tomorrow?", *out.Input[1].Content.ContentBlocks[0].Text)
}

func TestGeminiInteractionsRequestNativeOnlyFields(t *testing.T) {
	var req GeminiInteractionsRequest
	require.NoError(t, sonic.Unmarshal([]byte(`{"agent":"deep-research-preview-04-2026","input":"go","agent_config":{"type":"deep-research"},"background":true}`), &req))
	err := req.ValidateTranslatable()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent")
	assert.Contains(t, err.Error(), "agent_config")

	out, convErr := req.ToBifrostResponsesRequest()
	require.NoError(t, convErr)
	assert.Equal(t, "deep-research-preview-04-2026", out.Model)
	assert.JSONEq(t, `{"type":"deep-research"}`, string(out.Params.ExtraParams["agent_config"].(json.RawMessage)))
}

func TestGeminiInteractionToBifrostResponses(t *testing.T) {
	body := `{
		"id":"v1_run","object":"interaction","model":"gemini-3.8-flash","status":"requires_action",
		"created":"2026-11-18T10:00:00Z","updated":"2026-11-18T10:00:05Z",
		"steps":[
			{"type":"thought","signature":"sig_a","summary":[{"type":"text","text":"Thinking"}]},
			{"type":"google_search_call","id":"s1","arguments":{"queries":["paris weather"]}},
			{"type":"model_output","content":[{"type":"text","text":"Checking."}]},
			{"type":"function_call","id":"call_9","name":"get_weather","arguments":{"city":"Paris"}}
		],
		"usage":{"total_input_tokens":10,"total_output_tokens":5,"total_thought_tokens":7,"total_cached_tokens":2,"total_tokens":22}
	}`
	var interaction GeminiInteraction
	require.NoError(t, sonic.Unmarshal([]byte(body), &interaction))
	resp := interaction.ToBifrostResponsesResponse()

	assert.Equal(t, "v1_run", *resp.ID)
	assert.Equal(t, "gemini-3.8-flash", resp.Model)
	assert.Equal(t, schemas.ResponsesResponseStatusCompleted, *resp.Status, "requires_action is a finished turn waiting on function results")
	require.Len(t, resp.Output, 3)
	assert.Equal(t, schemas.ResponsesMessageTypeReasoning, *resp.Output[0].Type)
	assert.Equal(t, "sig_a", *resp.Output[0].ResponsesReasoning.EncryptedContent)
	assert.Equal(t, schemas.ResponsesMessageTypeMessage, *resp.Output[1].Type)
	assert.Equal(t, "Checking.", *resp.Output[1].Content.ContentBlocks[0].Text)
	assert.Equal(t, schemas.ResponsesMessageTypeFunctionCall, *resp.Output[2].Type)
	assert.Equal(t, "call_9", *resp.Output[2].CallID)
	assert.JSONEq(t, `{"city":"Paris"}`, *resp.Output[2].Arguments)
	assert.NotNil(t, resp.ProviderExtraFields["steps"], "built-in tool steps are kept")

	require.NotNil(t, resp.Usage)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 12, resp.Usage.OutputTokens, "thought tokens are output")
	assert.Equal(t, 7, resp.Usage.OutputTokensDetails.ReasoningTokens)
	assert.Equal(t, 2, resp.Usage.InputTokensDetails.CachedReadTokens)
	assert.Equal(t, 22, resp.Usage.TotalTokens)
}

func TestToGeminiInteractionFromResponses(t *testing.T) {
	resp := &schemas.BifrostResponsesResponse{
		ID: schemas.Ptr("resp_1"), Model: "gpt-5", CreatedAt: 1700000000, Status: schemas.Ptr("completed"),
		Output: []schemas.ResponsesMessage{
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant), Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("Calling")}}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("f"), Arguments: schemas.Ptr(`{"a":1}`)}},
		},
		Usage: &schemas.ResponsesResponseUsage{InputTokens: 3, OutputTokens: 9, TotalTokens: 12, OutputTokensDetails: &schemas.ResponsesResponseOutputTokens{ReasoningTokens: 4}},
	}
	out := ToGeminiInteraction(resp)
	raw, err := providerUtils.MarshalSorted(out)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"id":"resp_1","object":"interaction","model":"gpt-5","status":"requires_action",
		"created":"2023-11-14T22:13:20Z","updated":"2023-11-14T22:13:20Z",
		"steps":[
			{"type":"model_output","content":[{"type":"text","text":"Calling"}]},
			{"type":"function_call","id":"call_1","name":"f","arguments":{"a":1}}
		],
		"usage":{"total_input_tokens":3,"total_output_tokens":5,"total_thought_tokens":4,"total_tokens":12}
	}`, string(raw))
}

func interactionsEvents(t *testing.T, lines ...string) []*GeminiInteractionsEvent {
	t.Helper()
	events := make([]*GeminiInteractionsEvent, 0, len(lines))
	for _, line := range lines {
		event, err := ParseGeminiInteractionsEvent([]byte(line))
		require.NoError(t, err)
		events = append(events, event)
	}
	return events
}

// TestGeminiInteractionsStreamToBifrost pins the SSE mapping: steps open and close output items,
// text, thought summaries and argument fragments become their delta events, and the completed
// event carries the run's usage.
func TestGeminiInteractionsStreamToBifrost(t *testing.T) {
	events := interactionsEvents(t,
		`{"event_type":"interaction.created","interaction":{"id":"v1_s","status":"in_progress","model":"gemini-3.8-flash"}}`,
		`{"event_type":"interaction.status_update","interaction_id":"v1_s","status":"in_progress"}`,
		`{"event_type":"step.start","index":0,"step":{"type":"thought"}}`,
		`{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"Plan"}}}`,
		`{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig_z"}}`,
		`{"event_type":"step.stop","index":0}`,
		`{"event_type":"step.start","index":1,"step":{"type":"model_output"}}`,
		`{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"Hel"}}`,
		`{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"lo"}}`,
		`{"event_type":"step.stop","index":1}`,
		`{"event_type":"step.start","index":2,"step":{"type":"function_call","id":"call_5","name":"f","arguments":{}}}`,
		`{"event_type":"step.delta","index":2,"delta":{"type":"arguments_delta","arguments":"{\"a\":"}}`,
		`{"event_type":"step.delta","index":2,"delta":{"type":"arguments_delta","arguments":"1}"}}`,
		`{"event_type":"step.stop","index":2}`,
		`{"event_type":"interaction.completed","interaction":{"id":"v1_s","status":"requires_action","model":"gemini-3.8-flash","usage":{"total_input_tokens":4,"total_output_tokens":6,"total_thought_tokens":2,"total_tokens":12}}}`,
	)
	state := NewGeminiInteractionsStreamState("gemini-3.8-flash")
	var types []schemas.ResponsesStreamResponseType
	var last *schemas.BifrostResponsesStreamResponse
	perEvent := make([]int, len(events))
	for i, event := range events {
		out, err := state.ToBifrostResponsesStream(event)
		require.Nil(t, err)
		perEvent[i] = len(out)
		for _, ev := range out {
			types = append(types, ev.Type)
			last = ev
			if ev.Type == schemas.ResponsesStreamResponseTypeOutputItemDone && ev.Item != nil && ev.Item.Type != nil {
				switch *ev.Item.Type {
				case schemas.ResponsesMessageTypeReasoning:
					assert.Equal(t, "sig_z", *ev.Item.ResponsesReasoning.EncryptedContent)
					assert.Equal(t, "Plan", ev.Item.ResponsesReasoning.Summary[0].Text)
				case schemas.ResponsesMessageTypeMessage:
					assert.Equal(t, "Hello", *ev.Item.Content.ContentBlocks[0].Text)
				case schemas.ResponsesMessageTypeFunctionCall:
					assert.Equal(t, `{"a":1}`, *ev.Item.Arguments)
					assert.Equal(t, "call_5", *ev.Item.CallID)
				}
			}
		}
	}
	assert.Equal(t, 0, perEvent[1], "a status update between steps has no Responses event")
	assert.Equal(t, []schemas.ResponsesStreamResponseType{
		schemas.ResponsesStreamResponseTypeCreated, schemas.ResponsesStreamResponseTypeInProgress,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeReasoningSummaryPartAdded, schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
		schemas.ResponsesStreamResponseTypeReasoningSummaryTextDone, schemas.ResponsesStreamResponseTypeReasoningSummaryPartDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeContentPartAdded, schemas.ResponsesStreamResponseTypeOutputTextDelta, schemas.ResponsesStreamResponseTypeOutputTextDelta,
		schemas.ResponsesStreamResponseTypeOutputTextDone, schemas.ResponsesStreamResponseTypeContentPartDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeOutputItemAdded,
		schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta, schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
		schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDone, schemas.ResponsesStreamResponseTypeOutputItemDone,
		schemas.ResponsesStreamResponseTypeCompleted,
	}, types)
	require.NotNil(t, last.Response)
	assert.Equal(t, "v1_s", *last.Response.ID)
	assert.Equal(t, 8, last.Response.Usage.OutputTokens)
	assert.Equal(t, 12, last.Response.Usage.TotalTokens)
}

func TestGeminiInteractionsStreamError(t *testing.T) {
	state := NewGeminiInteractionsStreamState("m")
	_, err := state.ToBifrostResponsesStream(interactionsEvents(t, `{"event_type":"error","error":{"code":"resource_exhausted","message":"quota"}}`)[0])
	require.NotNil(t, err)
	assert.Equal(t, "quota", err.Error.Message)
	assert.Equal(t, "resource_exhausted", *err.Error.Code)
}

// TestBifrostToGeminiInteractionsStream pins the reverse rendering for a translated stream.
func TestBifrostToGeminiInteractionsStream(t *testing.T) {
	idx := 0
	state := NewBifrostToGeminiInteractionsStreamState()
	var got []string
	for _, ev := range []*schemas.BifrostResponsesStreamResponse{
		{Type: schemas.ResponsesStreamResponseTypeCreated, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_9"), Model: "gpt-5"}},
		{Type: schemas.ResponsesStreamResponseTypeInProgress},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemAdded, OutputIndex: &idx, Item: &schemas.ResponsesMessage{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant)}},
		{Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, OutputIndex: &idx, Delta: schemas.Ptr("Hi")},
		{Type: schemas.ResponsesStreamResponseTypeOutputItemDone, OutputIndex: &idx},
		{Type: schemas.ResponsesStreamResponseTypeCompleted, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_9"), Model: "gpt-5", Status: schemas.Ptr("completed"), Usage: &schemas.ResponsesResponseUsage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}}},
	} {
		for _, out := range state.ToGeminiInteractionsStreamEvents(ev) {
			raw, err := sonic.Marshal(out)
			require.NoError(t, err)
			got = append(got, string(raw))
		}
	}
	require.Len(t, got, 5)
	assert.JSONEq(t, `{"event_type":"interaction.created","event_id":"evt_1","interaction":{"id":"resp_9","object":"interaction","model":"gpt-5","status":"in_progress"}}`, got[0])
	assert.JSONEq(t, `{"event_type":"step.start","event_id":"evt_2","index":0,"step":{"type":"model_output"}}`, got[1])
	assert.JSONEq(t, `{"event_type":"step.delta","event_id":"evt_3","index":0,"delta":{"type":"text","text":"Hi"}}`, got[2])
	assert.JSONEq(t, `{"event_type":"step.stop","event_id":"evt_4","index":0}`, got[3])
	assert.Contains(t, got[4], `"event_type":"interaction.completed"`)
	assert.Contains(t, got[4], `"total_tokens":2`)
}

func TestNormalizeRawGeminiInteractionsBody(t *testing.T) {
	out, err := normalizeRawGeminiInteractionsBody([]byte(`{"model":"gemini/gemini-3.8-flash","input":"hi","fallbacks":["vertex/gemini-3.8-flash"]}`), "gemini-3.8-flash")
	require.Nil(t, err)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"hi"}`, string(out))

	out, err = normalizeRawGeminiInteractionsBody([]byte(`{"agent":"gemini/deep-research-preview-04-2026","input":"go"}`), "deep-research-preview-04-2026")
	require.Nil(t, err)
	assert.JSONEq(t, `{"agent":"deep-research-preview-04-2026","input":"go"}`, string(out))
}

// newInteractionsTestServer answers Interactions calls with body and records each request.
type interactionsCall struct {
	method, path, query, apiKey, body string
}

func newInteractionsTestServer(t *testing.T, contentType string, reply func(call interactionsCall) string) (*httptest.Server, *[]interactionsCall) {
	t.Helper()
	calls := &[]interactionsCall{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		call := interactionsCall{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, apiKey: r.Header.Get("x-goog-api-key"), body: string(raw)}
		*calls = append(*calls, call)
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, reply(call))
	}))
	t.Cleanup(server.Close)
	return server, calls
}

// TestGeminiResponsesPreviousResponseIDUsesInteractions pins the egress switch: a Responses request
// naming a previous response goes to POST {base}/interactions with previous_interaction_id, and the
// interaction id comes back as the response id.
func TestGeminiResponsesPreviousResponseIDUsesInteractions(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"id":"v1_next","object":"interaction","model":"gemini-3.8-flash","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Paris."}]}],"usage":{"total_input_tokens":5,"total_output_tokens":2,"total_tokens":7}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "test-key"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.Responses(ctx, key, &schemas.BifrostResponsesRequest{
		Provider: schemas.Gemini,
		Model:    "gemini-3.8-flash",
		Input:    []schemas.ResponsesMessage{interactionsUserText("And its capital?")},
		Params:   &schemas.ResponsesParameters{PreviousResponseID: schemas.Ptr("v1_prev")},
	})
	require.Nil(t, bifrostErr)
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/interactions", (*calls)[0].path)
	assert.Equal(t, "test-key", (*calls)[0].apiKey)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"And its capital?","previous_interaction_id":"v1_prev"}`, (*calls)[0].body)
	assert.Equal(t, "v1_next", *resp.ID)
	assert.Equal(t, "Paris.", *resp.Output[0].Content.ContentBlocks[0].Text)
	assert.Equal(t, 7, resp.Usage.TotalTokens)
	assert.Nil(t, resp.ProviderJob, "a run that finished in the foreground is billed on its own row")
}

// TestGeminiResponsesBackgroundRecordsJob pins billing for a background run: the create reports
// the run on ProviderJob so job accounting settles it once it ends.
func TestGeminiResponsesBackgroundRecordsJob(t *testing.T) {
	server, _ := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"id":"v1_bg","object":"interaction","agent":"deep-research-preview-04-2026","status":"in_progress"}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	resp, bifrostErr := provider.Responses(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), key, &schemas.BifrostResponsesRequest{
		Model:  "deep-research-preview-04-2026",
		Input:  []schemas.ResponsesMessage{interactionsUserText("Research X")},
		Params: &schemas.ResponsesParameters{Background: schemas.Ptr(true)},
	})
	require.Nil(t, bifrostErr)
	assert.Equal(t, schemas.ResponsesResponseStatusInProgress, *resp.Status)
	require.NotNil(t, resp.ProviderJob)
	assert.Equal(t, "v1_bg", resp.ProviderJob.ID)
	assert.True(t, resp.ProviderJob.Created)
	assert.False(t, resp.ProviderJob.Terminal)
	assert.Equal(t, "/interactions/v1_bg", resp.ProviderJob.PollPath)
}

// TestGeminiResponsesPlainRequestKeepsGenerateContent pins that the common case is untouched.
func TestGeminiResponsesPlainRequestKeepsGenerateContent(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(interactionsCall) string {
		return `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	_, bifrostErr := provider.Responses(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), key, &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{interactionsUserText("hi")},
	})
	require.Nil(t, bifrostErr)
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/models/gemini-3.8-flash:generateContent", (*calls)[0].path)
}

// TestGeminiResponsesLifecycle pins retrieve, cancel and delete on /interactions/{id}.
func TestGeminiResponsesLifecycle(t *testing.T) {
	server, calls := newInteractionsTestServer(t, "application/json", func(call interactionsCall) string {
		switch {
		case call.method == http.MethodDelete:
			return `{}`
		case strings.HasSuffix(call.path, "/cancel"):
			return `{"id":"v1_x","status":"cancelled"}`
		}
		return `{"id":"v1_x","status":"completed","model":"gemini-3.8-flash","steps":[{"type":"user_input","content":[{"type":"text","text":"q"}]},{"type":"model_output","content":[{"type":"text","text":"a"}]}],"usage":{"total_input_tokens":1,"total_output_tokens":1,"total_tokens":2}}`
	})
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	got, bifrostErr := provider.ResponsesRetrieve(ctx, key, &schemas.BifrostResponsesRetrieveRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	require.Len(t, got.Output, 1, "user_input steps are input, not output")
	require.NotNil(t, got.ProviderJob)
	assert.False(t, got.ProviderJob.Created)
	assert.True(t, got.ProviderJob.Terminal)
	assert.Equal(t, 2, got.ProviderJob.Usage.LLMUsage.TotalTokens)

	cancelled, bifrostErr := provider.ResponsesCancel(ctx, key, &schemas.BifrostResponsesCancelRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	assert.Equal(t, schemas.ResponsesResponseStatusCancelled, *cancelled.Status)

	deleted, bifrostErr := provider.ResponsesDelete(ctx, key, &schemas.BifrostResponsesDeleteRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	assert.True(t, deleted.Deleted)

	items, bifrostErr := provider.ResponsesInputItems(ctx, key, &schemas.BifrostResponsesInputItemsRequest{ResponseID: "v1_x"})
	require.Nil(t, bifrostErr)
	require.Len(t, items.Data, 1)
	assert.Equal(t, "q", *items.Data[0].Content.ContentBlocks[0].Text)

	require.Len(t, *calls, 4)
	assert.Equal(t, "GET /v1beta/interactions/v1_x", (*calls)[0].method+" "+(*calls)[0].path)
	assert.Equal(t, "POST /v1beta/interactions/v1_x/cancel", (*calls)[1].method+" "+(*calls)[1].path)
	assert.Equal(t, "DELETE /v1beta/interactions/v1_x", (*calls)[2].method+" "+(*calls)[2].path)
}

// TestGeminiResponsesStreamViaInteractions pins the streamed create: stream:true in the body,
// alt=sse on the URL, Google's SSE framing parsed, and each raw event attached to one chunk for
// the native route (an event with no Responses counterpart rides a raw-only chunk).
func TestGeminiResponsesStreamViaInteractions(t *testing.T) {
	sse := strings.Join([]string{
		"event: interaction.created", `data: {"event_type":"interaction.created","interaction":{"id":"v1_s","status":"in_progress","model":"gemini-3.8-flash"}}`, "",
		"event: interaction.status_update", `data: {"event_type":"interaction.status_update","interaction_id":"v1_s","status":"in_progress"}`, "",
		"event: step.start", `data: {"event_type":"step.start","index":0,"step":{"type":"model_output"}}`, "",
		"event: step.delta", `data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hi"}}`, "",
		"event: step.stop", `data: {"event_type":"step.stop","index":0}`, "",
		"event: interaction.completed", `data: {"event_type":"interaction.completed","interaction":{"id":"v1_s","status":"completed","usage":{"total_input_tokens":1,"total_output_tokens":1,"total_tokens":2}}}`, "",
		"event: done", "data: [DONE]", "",
	}, "\n")
	server, calls := newInteractionsTestServer(t, "text/event-stream", func(interactionsCall) string { return sse })
	provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL + "/v1beta"}}, nil)
	key := schemas.Key{Value: schemas.SecretVar{Val: "k"}, Models: []string{"*"}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyInteractionsAPI, true)
	ctx.SetValue(schemas.BifrostContextKeyCaptureRawResponse, true) // what core derives from the route's send-back-raw flag

	postHook := func(ctx *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return result, err
	}
	stream, bifrostErr := provider.ResponsesStream(ctx, postHook, nil, key, &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Input: []schemas.ResponsesMessage{interactionsUserText("hi")}})
	require.Nil(t, bifrostErr)

	var rawEvents []string
	var types []schemas.ResponsesStreamResponseType
	for chunk := range stream {
		require.Nil(t, chunk.BifrostError)
		resp := chunk.BifrostResponsesStreamResponse
		require.NotNil(t, resp)
		types = append(types, resp.Type)
		if raw, ok := resp.ExtraFields.RawResponse.(string); ok {
			rawEvents = append(rawEvents, raw)
		}
	}
	require.Len(t, *calls, 1)
	assert.Equal(t, "/v1beta/interactions", (*calls)[0].path)
	assert.Equal(t, "alt=sse", (*calls)[0].query)
	assert.JSONEq(t, `{"model":"gemini-3.8-flash","input":"hi","stream":true}`, (*calls)[0].body)
	assert.Len(t, rawEvents, 6, "every upstream event reaches a native caller exactly once")
	assert.Contains(t, types, schemas.ResponsesStreamResponseTypeProviderRawEvent)
	assert.Equal(t, schemas.ResponsesStreamResponseTypeCompleted, types[len(types)-1])
}

// BenchmarkToGeminiInteractionsRequest measures converting a multi-turn Responses request with a
// tool into an Interactions body, the work the /v1/responses -> Gemini Interactions path adds.
func BenchmarkToGeminiInteractionsRequest(b *testing.B) {
	req := &schemas.BifrostResponsesRequest{
		Model: "gemini-3.8-flash",
		Input: []schemas.ResponsesMessage{
			interactionsUserText("Weather in Paris?"),
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Name: schemas.Ptr("get_weather"), Arguments: schemas.Ptr(`{"city":"Paris"}`)}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("call_1"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr(`{"temp":21}`)}}},
		},
		Params: &schemas.ResponsesParameters{
			PreviousResponseID: schemas.Ptr("v1_prev"),
			Tools:              []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("get_weather"), ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &schemas.ToolFunctionParameters{Type: "object"}}}},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		native, err := ToGeminiInteractionsRequest(req)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := providerUtils.MarshalSorted(native); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGeminiInteractionsStreamEvent measures parsing and converting one text delta event, the
// per-chunk cost of an Interactions stream.
func BenchmarkGeminiInteractionsStreamEvent(b *testing.B) {
	data := []byte(`{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hello there"}}`)
	state := NewGeminiInteractionsStreamState("gemini-3.8-flash")
	start, _ := ParseGeminiInteractionsEvent([]byte(`{"event_type":"step.start","index":0,"step":{"type":"model_output"}}`))
	_, _ = state.ToBifrostResponsesStream(start)
	b.ReportAllocs()
	for b.Loop() {
		event, err := ParseGeminiInteractionsEvent(data)
		if err != nil {
			b.Fatal(err)
		}
		_, _ = state.ToBifrostResponsesStream(event)
	}
}

// BenchmarkShouldUseGeminiInteractions measures the gate every Gemini and Vertex Responses request
// now passes, for the common case that stays on generateContent.
func BenchmarkShouldUseGeminiInteractions(b *testing.B) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	req := &schemas.BifrostResponsesRequest{Model: "gemini-3.8-flash", Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(100)}}
	b.ReportAllocs()
	for b.Loop() {
		if ShouldUseGeminiInteractions(ctx, req) {
			b.Fatal("plain request left generateContent")
		}
	}
}
