use std::convert::TryFrom;

use async_stream::try_stream;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info_span;

use super::client::Client;
use super::message::{Message, ToolDefinition};
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage, Usage};
use crate::http_client::{self, HttpClientExt};
use crate::streaming::RawStreamingChoice;
use crate::{OneOrMany, json_utils, message, streaming};

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
	pub model: String,
	pub created_at: String,
	pub message: Message,
	pub done: bool,
	#[serde(default)]
	pub done_reason: Option<String>,
	#[serde(default)]
	pub total_duration: Option<u64>,
	#[serde(default)]
	pub load_duration: Option<u64>,
	#[serde(default)]
	pub prompt_eval_count: Option<u64>,
	#[serde(default)]
	pub prompt_eval_duration: Option<u64>,
	#[serde(default)]
	pub eval_count: Option<u64>,
	#[serde(default)]
	pub eval_duration: Option<u64>,
}
impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
	type Error = CompletionError;
	fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
		match resp.message {
			// Process only if an assistant message is present.
			Message::Assistant {
				content,
				thinking,
				tool_calls,
				..
			} => {
				let mut assistant_contents = Vec::new();
				// Add the assistant's text content if any.
				if !content.is_empty() {
					assistant_contents.push(completion::AssistantContent::text(&content));
				}
				// Process tool_calls following Ollama's chat response definition.
				// Each ToolCall has an id, a type, and a function field.
				for tc in tool_calls.iter() {
					assistant_contents.push(completion::AssistantContent::tool_call(
						tc.function.name.clone(),
						tc.function.name.clone(),
						tc.function.arguments.clone(),
					));
				}
				let choice = OneOrMany::many(assistant_contents).map_err(|_| {
					CompletionError::ResponseError("No content provided".to_owned())
				})?;
				let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
				let completion_tokens = resp.eval_count.unwrap_or(0);

				let raw_response = CompletionResponse {
					model: resp.model,
					created_at: resp.created_at,
					done: resp.done,
					done_reason: resp.done_reason,
					total_duration: resp.total_duration,
					load_duration: resp.load_duration,
					prompt_eval_count: resp.prompt_eval_count,
					prompt_eval_duration: resp.prompt_eval_duration,
					eval_count: resp.eval_count,
					eval_duration: resp.eval_duration,
					message: Message::Assistant {
						content,
						thinking,
						images: None,
						name: None,
						tool_calls,
					},
				};

				Ok(completion::CompletionResponse {
					choice,
					usage: Usage {
						input_tokens: prompt_tokens,
						output_tokens: completion_tokens,
						total_tokens: prompt_tokens + completion_tokens,
						cached_input_tokens: 0,
					},
					raw_response,
				})
			}
			_ => Err(CompletionError::ResponseError(
				"Chat response does not include an assistant message".into(),
			)),
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OllamaCompletionRequest {
	model: String,
	pub messages: Vec<Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<ToolDefinition>,
	pub stream: bool,
	think: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_tokens: Option<u64>,
	options: serde_json::Value,
}

impl TryFrom<(&str, CompletionRequest)> for OllamaCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		if req.tool_choice.is_some() {
			tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
		}
		// Build up the order of messages (context, chat_history, prompt)
		let mut partial_history = vec![];
		if let Some(docs) = req.normalized_documents() {
			partial_history.push(docs);
		}
		partial_history.extend(req.chat_history);

		// Add preamble to chat history (if available)
		let mut full_history: Vec<Message> = match &req.preamble {
			Some(preamble) => vec![Message::system(preamble)],
			None => vec![],
		};

		// Convert and extend the rest of the history
		full_history.extend(
			partial_history
				.into_iter()
				.map(message::Message::try_into)
				.collect::<Result<Vec<Vec<Message>>, _>>()?
				.into_iter()
				.flatten()
				.collect::<Vec<_>>(),
		);

		let mut think = false;

		// TODO: Fix this up to include the full range of ollama options
		let options = if let Some(mut extra) = req.additional_params {
			if extra.get("think").is_some() {
				think = extra["think"].take().as_bool().ok_or_else(|| {
					CompletionError::RequestError("`think` must be a bool".into())
				})?;
			}
			json_utils::merge(json!({ "temperature": req.temperature }), extra)
		} else {
			json!({ "temperature": req.temperature })
		};

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			max_tokens: req.max_tokens,
			stream: false,
			think,
			tools: req
				.tools
				.clone()
				.into_iter()
				.map(ToolDefinition::from)
				.collect::<Vec<_>>(),
			options,
		})
	}
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
	client: Client<T>,
	pub model: String,
}

impl<T> CompletionModel<T> {
	pub fn new(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.to_owned(),
		}
	}
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
	pub done_reason: Option<String>,
	pub total_duration: Option<u64>,
	pub load_duration: Option<u64>,
	pub prompt_eval_count: Option<u64>,
	pub prompt_eval_duration: Option<u64>,
	pub eval_count: Option<u64>,
	pub eval_duration: Option<u64>,
}

impl GetTokenUsage for StreamingCompletionResponse {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();
		let input_tokens = self.prompt_eval_count.unwrap_or_default();
		let output_tokens = self.eval_count.unwrap_or_default();
		usage.input_tokens = input_tokens;
		usage.output_tokens = output_tokens;
		usage.total_tokens = input_tokens + output_tokens;

		Some(usage)
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model.into().as_str())
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "ollama",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = tracing::field::Empty,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		span.record("gen_ai.system_instructions", &completion_request.preamble);
		let request = OllamaCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Ollama completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("api/chat")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let async_block = async move {
			let response = self.client.send::<_, Bytes>(req).await?;
			let status = response.status();
			let response_body = response.into_body().into_future().await?.to_vec();

			if !status.is_success() {
				return Err(CompletionError::ProviderError(
					String::from_utf8_lossy(&response_body).to_string(),
				));
			}

			let response: CompletionResponse = serde_json::from_slice(&response_body)?;
			let span = tracing::Span::current();
			span.record("gen_ai.response.model_name", &response.model);
			span.record(
				"gen_ai.usage.input_tokens",
				response.prompt_eval_count.unwrap_or_default(),
			);
			span.record(
				"gen_ai.usage.output_tokens",
				response.eval_count.unwrap_or_default(),
			);

			if tracing::enabled!(tracing::Level::TRACE) {
				tracing::trace!(target: "clankers::completions",
					"Ollama completion response: {}",
					serde_json::to_string_pretty(&response)?
				);
			}

			let response: completion::CompletionResponse<CompletionResponse> =
				response.try_into()?;

			Ok(response)
		};

		tracing::Instrument::instrument(async_block, span).await
	}

	async fn stream(
		&self,
		request: CompletionRequest,
	) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat_streaming",
				gen_ai.operation.name = "chat_streaming",
				gen_ai.provider.name = "ollama",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = tracing::field::Empty,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = self.model,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		span.record("gen_ai.system_instructions", &request.preamble);

		let mut request = OllamaCompletionRequest::try_from((self.model.as_ref(), request))?;
		request.stream = true;

		if tracing::enabled!(tracing::Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"Ollama streaming completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("api/chat")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let response = self.client.send_streaming(req).await?;
		let status = response.status();
		let mut byte_stream = response.into_body();

		if !status.is_success() {
			return Err(CompletionError::ProviderError(format!(
				"Got error status code trying to send a request to Ollama: {status}"
			)));
		}

		let stream = try_stream! {
            let span = tracing::Span::current();
            let mut tool_calls_final = Vec::new();
            let mut text_response = String::new();
            let mut thinking_response = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| http_client::Error::Instance(e.into()))?;

                for line in bytes.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }

                    tracing::debug!(target: "clankers", "Received NDJSON line from Ollama: {}", String::from_utf8_lossy(line));

                    let response: CompletionResponse = serde_json::from_slice(line)?;

                    if let Message::Assistant { content, thinking, tool_calls, .. } = response.message {
                        if let Some(thinking_content) = thinking && !thinking_content.is_empty() {
                            thinking_response += &thinking_content;
                            yield RawStreamingChoice::ReasoningDelta {
                                id: None,
                                reasoning: thinking_content,
                            };
                        }

                        if !content.is_empty() {
                            text_response += &content;
                            yield RawStreamingChoice::Message(content);
                        }

                        for tool_call in tool_calls {
                            tool_calls_final.push(tool_call.clone());
                            yield RawStreamingChoice::ToolCall(
                                crate::streaming::RawStreamingToolCall::new(String::new(), tool_call.function.name, tool_call.function.arguments)
                            );
                        }
                    }

                    if response.done {
                        span.record("gen_ai.usage.input_tokens", response.prompt_eval_count);
                        span.record("gen_ai.usage.output_tokens", response.eval_count);
                        let message = Message::Assistant {
                            content: text_response.clone(),
                            thinking: if thinking_response.is_empty() { None } else { Some(thinking_response.clone()) },
                            images: None,
                            name: None,
                            tool_calls: tool_calls_final.clone()
                        };
                        span.record("gen_ai.output.messages", serde_json::to_string(&vec![message]).unwrap());
                        yield RawStreamingChoice::FinalResponse(
                            StreamingCompletionResponse {
                                total_duration: response.total_duration,
                                load_duration: response.load_duration,
                                prompt_eval_count: response.prompt_eval_count,
                                prompt_eval_duration: response.prompt_eval_duration,
                                eval_count: response.eval_count,
                                eval_duration: response.eval_duration,
                                done_reason: response.done_reason,
                            }
                        );
                        break;
                    }
                }
            }
        }.instrument(span);

		Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
			stream,
		)))
	}
}

use tracing_futures::Instrument;

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	// Test deserialization and conversion for the /api/chat endpoint.
	#[tokio::test]
	async fn test_chat_completion() {
		// Sample JSON response from /api/chat (non-streaming) based on Ollama docs.
		let sample_chat_response = json!({
			"model": "llama3.2",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "The sky is blue because of Rayleigh scattering.",
				"images": null,
				"tool_calls": [
					{
						"type": "function",
						"function": {
							"name": "get_current_weather",
							"arguments": {
								"location": "San Francisco, CA",
								"format": "celsius"
							}
						}
					}
				]
			},
			"done": true,
			"total_duration": 8000000000u64,
			"load_duration": 6000000u64,
			"prompt_eval_count": 61u64,
			"prompt_eval_duration": 400000000u64,
			"eval_count": 468u64,
			"eval_duration": 7700000000u64
		});
		let sample_text = sample_chat_response.to_string();

		let chat_resp: CompletionResponse =
			serde_json::from_str(&sample_text).expect("Invalid JSON structure");
		let conv: completion::CompletionResponse<CompletionResponse> =
			chat_resp.try_into().unwrap();
		assert!(
			!conv.choice.is_empty(),
			"Expected non-empty choice in chat response"
		);
	}

	// Test deserialization of chat response with thinking content
	#[tokio::test]
	async fn test_chat_completion_with_thinking() {
		let sample_response = json!({
			"model": "qwen-thinking",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "The answer is 42.",
				"thinking": "Let me think about this carefully. The question asks for the meaning of life...",
				"images": null,
				"tool_calls": []
			},
			"done": true,
			"total_duration": 8000000000u64,
			"load_duration": 6000000u64,
			"prompt_eval_count": 61u64,
			"prompt_eval_duration": 400000000u64,
			"eval_count": 468u64,
			"eval_duration": 7700000000u64
		});

		let chat_resp: CompletionResponse =
			serde_json::from_value(sample_response).expect("Failed to deserialize");

		// Verify thinking field is present
		if let Message::Assistant {
			thinking, content, ..
		} = &chat_resp.message
		{
			assert_eq!(
				thinking.as_ref().unwrap(),
				"Let me think about this carefully. The question asks for the meaning of life..."
			);
			assert_eq!(content, "The answer is 42.");
		} else {
			panic!("Expected Assistant message");
		}
	}

	// Test deserialization of chat response without thinking content
	#[tokio::test]
	async fn test_chat_completion_without_thinking() {
		let sample_response = json!({
			"model": "llama3.2",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "Hello!",
				"images": null,
				"tool_calls": []
			},
			"done": true,
			"total_duration": 8000000000u64,
			"load_duration": 6000000u64,
			"prompt_eval_count": 10u64,
			"prompt_eval_duration": 400000000u64,
			"eval_count": 5u64,
			"eval_duration": 7700000000u64
		});

		let chat_resp: CompletionResponse =
			serde_json::from_value(sample_response).expect("Failed to deserialize");

		// Verify thinking field is None when not provided
		if let Message::Assistant {
			thinking, content, ..
		} = &chat_resp.message
		{
			assert!(thinking.is_none());
			assert_eq!(content, "Hello!");
		} else {
			panic!("Expected Assistant message");
		}
	}

	// Test deserialization of streaming response with thinking content
	#[test]
	fn test_streaming_response_with_thinking() {
		let sample_chunk = json!({
			"model": "qwen-thinking",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "",
				"thinking": "Analyzing the problem...",
				"images": null,
				"tool_calls": []
			},
			"done": false
		});

		let chunk: CompletionResponse =
			serde_json::from_value(sample_chunk).expect("Failed to deserialize");

		if let Message::Assistant {
			thinking, content, ..
		} = &chunk.message
		{
			assert_eq!(thinking.as_ref().unwrap(), "Analyzing the problem...");
			assert_eq!(content, "");
		} else {
			panic!("Expected Assistant message");
		}
	}

	// Test empty thinking content is handled correctly
	#[test]
	fn test_empty_thinking_content() {
		let sample_response = json!({
			"model": "llama3.2",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "Response",
				"thinking": "",
				"images": null,
				"tool_calls": []
			},
			"done": true,
			"total_duration": 8000000000u64,
			"load_duration": 6000000u64,
			"prompt_eval_count": 10u64,
			"prompt_eval_duration": 400000000u64,
			"eval_count": 5u64,
			"eval_duration": 7700000000u64
		});

		let chat_resp: CompletionResponse =
			serde_json::from_value(sample_response).expect("Failed to deserialize");

		if let Message::Assistant {
			thinking, content, ..
		} = &chat_resp.message
		{
			// Empty string should still deserialize as Some("")
			assert_eq!(thinking.as_ref().unwrap(), "");
			assert_eq!(content, "Response");
		} else {
			panic!("Expected Assistant message");
		}
	}

	// Test thinking with tool calls
	#[test]
	fn test_thinking_with_tool_calls() {
		let sample_response = json!({
			"model": "qwen-thinking",
			"created_at": "2023-08-04T19:22:45.499127Z",
			"message": {
				"role": "assistant",
				"content": "Let me check the weather.",
				"thinking": "User wants weather info, I should use the weather tool",
				"images": null,
				"tool_calls": [
					{
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": {
								"location": "San Francisco"
							}
						}
					}
				]
			},
			"done": true,
			"total_duration": 8000000000u64,
			"load_duration": 6000000u64,
			"prompt_eval_count": 30u64,
			"prompt_eval_duration": 400000000u64,
			"eval_count": 50u64,
			"eval_duration": 7700000000u64
		});

		let chat_resp: CompletionResponse =
			serde_json::from_value(sample_response).expect("Failed to deserialize");

		if let Message::Assistant {
			thinking,
			content,
			tool_calls,
			..
		} = &chat_resp.message
		{
			assert_eq!(
				thinking.as_ref().unwrap(),
				"User wants weather info, I should use the weather tool"
			);
			assert_eq!(content, "Let me check the weather.");
			assert_eq!(tool_calls.len(), 1);
			assert_eq!(tool_calls[0].function.name, "get_weather");
		} else {
			panic!("Expected Assistant message with thinking and tool calls");
		}
	}
}
