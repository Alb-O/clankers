use std::convert::Infallible;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::message::{self};
use crate::one_or_many::string_or_one_or_many;
use crate::{OneOrMany, json_utils};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
	Ok(T),
	Err(Value),
}

// Conversational LLMs
/// `google/gemma-2-2b-it` completion model
pub const GEMMA_2: &str = "google/gemma-2-2b-it";
/// `meta-llama/Meta-Llama-3.1-8B-Instruct` completion model
pub const META_LLAMA_3_1: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// `PowerInfer/SmallThinker-3B-Preview` completion model
pub const SMALLTHINKER_PREVIEW: &str = "PowerInfer/SmallThinker-3B-Preview";
/// `Qwen/Qwen2.5-7B-Instruct` completion model
pub const QWEN2_5: &str = "Qwen/Qwen2.5-7B-Instruct";
/// `Qwen/Qwen2.5-Coder-32B-Instruct` completion model
pub const QWEN2_5_CODER: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";

// Conversational VLMs

/// `Qwen/Qwen2-VL-7B-Instruct` visual-language completion model
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-7B-Instruct";
/// `Qwen/QVQ-72B-Preview` visual-language completion model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct Function {
	pub name: String,
	#[serde(
		serialize_with = "json_utils::stringified_json::serialize",
		deserialize_with = "deserialize_arguments"
	)]
	pub arguments: serde_json::Value,
}

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
	D: Deserializer<'de>,
{
	let value = Value::deserialize(deserializer)?;

	match value {
		Value::String(s) => serde_json::from_str(&s).map_err(serde::de::Error::custom),
		other => Ok(other),
	}
}

impl From<Function> for message::ToolFunction {
	fn from(value: Function) -> Self {
		message::ToolFunction {
			name: value.name,
			arguments: value.arguments,
		}
	}
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
	#[default]
	Function,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
	pub r#type: String,
	pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
	fn from(tool: completion::ToolDefinition) -> Self {
		Self {
			r#type: "function".into(),
			function: tool,
		}
	}
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct ToolCall {
	pub id: String,
	pub r#type: ToolType,
	pub function: Function,
}

impl From<ToolCall> for message::ToolCall {
	fn from(value: ToolCall) -> Self {
		message::ToolCall {
			id: value.id,
			call_id: None,
			function: value.function.into(),
			signature: None,
			additional_params: None,
		}
	}
}

impl From<message::ToolCall> for ToolCall {
	fn from(value: message::ToolCall) -> Self {
		ToolCall {
			id: value.id,
			r#type: ToolType::Function,
			function: Function {
				name: value.function.name,
				arguments: value.function.arguments,
			},
		}
	}
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct ImageUrl {
	pub url: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
	Text {
		text: String,
	},
	#[serde(rename = "image_url")]
	ImageUrl {
		image_url: ImageUrl,
	},
}

impl FromStr for UserContent {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(UserContent::Text {
			text: s.to_string(),
		})
	}
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
	Text { text: String },
}

impl FromStr for AssistantContent {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(AssistantContent::Text {
			text: s.to_string(),
		})
	}
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemContent {
	Text { text: String },
}

impl FromStr for SystemContent {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(SystemContent::Text {
			text: s.to_string(),
		})
	}
}

impl From<UserContent> for message::UserContent {
	fn from(value: UserContent) -> Self {
		match value {
			UserContent::Text { text } => message::UserContent::text(text),
			UserContent::ImageUrl { image_url } => {
				message::UserContent::image_url(image_url.url, None, None)
			}
		}
	}
}

impl TryFrom<message::UserContent> for UserContent {
	type Error = message::MessageError;

	fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
		match content {
			message::UserContent::Text(text) => Ok(UserContent::Text { text: text.text }),
			message::UserContent::Document(message::Document {
				data: message::DocumentSourceKind::Raw(raw),
				..
			}) => {
				let text = String::from_utf8_lossy(raw.as_slice()).into();
				Ok(UserContent::Text { text })
			}
			message::UserContent::Document(message::Document {
				data:
					message::DocumentSourceKind::Base64(text)
					| message::DocumentSourceKind::String(text),
				..
			}) => Ok(UserContent::Text { text }),
			message::UserContent::Image(message::Image { data, .. }) => match data {
				message::DocumentSourceKind::Url(url) => Ok(UserContent::ImageUrl {
					image_url: ImageUrl { url },
				}),
				_ => Err(message::MessageError::ConversionError(
					"Huggingface only supports images as urls".into(),
				)),
			},
			_ => Err(message::MessageError::ConversionError(
				"Huggingface only supports text and images".into(),
			)),
		}
	}
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
	System {
		#[serde(deserialize_with = "string_or_one_or_many")]
		content: OneOrMany<SystemContent>,
	},
	User {
		#[serde(deserialize_with = "string_or_one_or_many")]
		content: OneOrMany<UserContent>,
	},
	Assistant {
		#[serde(default, deserialize_with = "json_utils::string_or_vec")]
		content: Vec<AssistantContent>,
		#[serde(default, deserialize_with = "json_utils::null_or_vec")]
		tool_calls: Vec<ToolCall>,
	},
	#[serde(rename = "tool", alias = "Tool")]
	ToolResult {
		name: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		arguments: Option<serde_json::Value>,
		#[serde(
			deserialize_with = "string_or_one_or_many",
			serialize_with = "serialize_tool_content"
		)]
		content: OneOrMany<String>,
	},
}

fn serialize_tool_content<S>(content: &OneOrMany<String>, serializer: S) -> Result<S::Ok, S::Error>
where
	S: Serializer,
{
	// OpenAI-compatible APIs expect tool content as a string, not an array
	let joined = content
		.iter()
		.map(String::as_str)
		.collect::<Vec<_>>()
		.join("\n");
	serializer.serialize_str(&joined)
}

impl Message {
	pub fn system(content: &str) -> Self {
		Message::System {
			content: OneOrMany::one(SystemContent::Text {
				text: content.to_string(),
			}),
		}
	}
}

impl TryFrom<message::Message> for Vec<Message> {
	type Error = message::MessageError;

	fn try_from(message: message::Message) -> Result<Vec<Message>, Self::Error> {
		match message {
			message::Message::User { content } => {
				let (tool_results, other_content): (Vec<_>, Vec<_>) = content
					.into_iter()
					.partition(|content| matches!(content, message::UserContent::ToolResult(_)));

				if !tool_results.is_empty() {
					tool_results
						.into_iter()
						.map(|content| match content {
							message::UserContent::ToolResult(message::ToolResult {
								id,
								content,
								..
							}) => Ok::<_, message::MessageError>(Message::ToolResult {
								name: id,
								arguments: None,
								content: content.try_map(|content| match content {
									message::ToolResultContent::Text(message::Text { text }) => {
										Ok(text)
									}
									_ => Err(message::MessageError::ConversionError(
										"Tool result content does not support non-text".into(),
									)),
								})?,
							}),
							_ => unreachable!(),
						})
						.collect::<Result<Vec<_>, _>>()
				} else {
					let other_content = OneOrMany::many(other_content).expect(
						"There must be other content here if there were no tool result content",
					);

					Ok(vec![Message::User {
                        content: other_content.try_map(|content| match content {
                            message::UserContent::Text(text) => {
                                Ok(UserContent::Text { text: text.text })
                            }
                            message::UserContent::Image(image) => {
                                let url = image.try_into_url()?;

                                Ok(UserContent::ImageUrl {
                                    image_url: ImageUrl { url },
                                })
                            }
                            message::UserContent::Document(message::Document {
                                data: message::DocumentSourceKind::Raw(raw), ..
                            }) => {
                                let text = String::from_utf8_lossy(raw.as_slice()).into();
                                Ok(UserContent::Text { text })
                            }
                            message::UserContent::Document(message::Document {
                                data: message::DocumentSourceKind::Base64(text) | message::DocumentSourceKind::String(text), ..
                            }) => {
                                Ok(UserContent::Text { text })
                            }
                            _ => Err(message::MessageError::ConversionError(
                                "Huggingface inputs only support text and image URLs (both base64-encoded images and regular URLs)".into(),
                            )),
                        })?,
                    }])
				}
			}
			message::Message::Assistant { content, .. } => {
				let (text_content, tool_calls) = content.into_iter().fold(
					(Vec::new(), Vec::new()),
					|(mut texts, mut tools), content| {
						match content {
							message::AssistantContent::Text(text) => texts.push(text),
							message::AssistantContent::ToolCall(tool_call) => tools.push(tool_call),
							message::AssistantContent::Reasoning(_) => {
								panic!("Reasoning is not supported on HuggingFace via Clankers");
							}
							message::AssistantContent::Image(_) => {
								panic!(
									"Image content is not supported on HuggingFace via Clankers"
								);
							}
						}
						(texts, tools)
					},
				);

				// `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
				//  so either `content` or `tool_calls` will have some content.
				Ok(vec![Message::Assistant {
					content: text_content
						.into_iter()
						.map(|content| AssistantContent::Text { text: content.text })
						.collect::<Vec<_>>(),
					tool_calls: tool_calls
						.into_iter()
						.map(|tool_call| tool_call.into())
						.collect::<Vec<_>>(),
				}])
			}
		}
	}
}

impl TryFrom<Message> for message::Message {
	type Error = message::MessageError;

	fn try_from(message: Message) -> Result<Self, Self::Error> {
		Ok(match message {
			Message::User { content, .. } => message::Message::User {
				content: content.map(|content| content.into()),
			},
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				let mut content = content
					.into_iter()
					.map(|content| match content {
						AssistantContent::Text { text } => message::AssistantContent::text(text),
					})
					.collect::<Vec<_>>();

				content.extend(
					tool_calls
						.into_iter()
						.map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
						.collect::<Result<Vec<_>, _>>()?,
				);

				message::Message::Assistant {
					id: None,
					content: OneOrMany::many(content).map_err(|_| {
						message::MessageError::ConversionError(
							"Neither `content` nor `tool_calls` was provided to the Message"
								.to_owned(),
						)
					})?,
				}
			}

			Message::ToolResult { name, content, .. } => message::Message::User {
				content: OneOrMany::one(message::UserContent::tool_result(
					name,
					content.map(message::ToolResultContent::text),
				)),
			},

			// System messages should get stripped out when converting message's, this is just a
			// stop gap to avoid obnoxious error handling or panic occurring.
			Message::System { content, .. } => message::Message::User {
				content: content.map(|c| match c {
					SystemContent::Text { text } => message::UserContent::text(text),
				}),
			},
		})
	}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Choice {
	pub finish_reason: String,
	pub index: usize,
	#[serde(default)]
	pub logprobs: serde_json::Value,
	pub message: Message,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Usage {
	pub completion_tokens: i32,
	pub prompt_tokens: i32,
	pub total_tokens: i32,
}

impl GetTokenUsage for Usage {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();
		usage.input_tokens = self.prompt_tokens as u64;
		usage.output_tokens = self.completion_tokens as u64;
		usage.total_tokens = self.total_tokens as u64;

		Some(usage)
	}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
	pub created: i32,
	pub id: String,
	pub model: String,
	pub choices: Vec<Choice>,
	#[serde(default, deserialize_with = "default_string_on_null")]
	pub system_fingerprint: String,
	pub usage: Usage,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
	type OutputMessage = Choice;
	type Usage = Usage;

	fn get_response_id(&self) -> Option<String> {
		Some(self.id.clone())
	}

	fn get_response_model_name(&self) -> Option<String> {
		Some(self.model.clone())
	}

	fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
		self.choices.clone()
	}

	fn get_text_response(&self) -> Option<String> {
		let text_response = self
			.choices
			.iter()
			.filter_map(|x| {
				let Message::User { ref content } = x.message else {
					return None;
				};

				let text = content
					.iter()
					.filter_map(|x| {
						if let UserContent::Text { text } = x {
							Some(text.clone())
						} else {
							None
						}
					})
					.collect::<Vec<String>>();

				if text.is_empty() {
					None
				} else {
					Some(text.join("\n"))
				}
			})
			.collect::<Vec<String>>()
			.join("\n");

		if text_response.is_empty() {
			None
		} else {
			Some(text_response)
		}
	}

	fn get_usage(&self) -> Option<Self::Usage> {
		Some(self.usage.clone())
	}
}

fn default_string_on_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
	D: Deserializer<'de>,
{
	match Option::<String>::deserialize(deserializer)? {
		Some(value) => Ok(value),      // Use provided value
		None => Ok(String::default()), // Use `Default` implementation
	}
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
	type Error = CompletionError;

	fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
		let choice = response.choices.first().ok_or_else(|| {
			CompletionError::ResponseError("Response contained no choices".to_owned())
		})?;

		let content = match &choice.message {
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				let mut content = content
					.iter()
					.map(|c| match c {
						AssistantContent::Text { text } => message::AssistantContent::text(text),
					})
					.collect::<Vec<_>>();

				content.extend(
					tool_calls
						.iter()
						.map(|call| {
							completion::AssistantContent::tool_call(
								&call.id,
								&call.function.name,
								call.function.arguments.clone(),
							)
						})
						.collect::<Vec<_>>(),
				);
				Ok(content)
			}
			_ => Err(CompletionError::ResponseError(
				"Response did not contain a valid message or tool call".into(),
			)),
		}?;

		let choice = OneOrMany::many(content).map_err(|_| {
			CompletionError::ResponseError(
				"Response contained no message or tool call (empty)".to_owned(),
			)
		})?;

		let usage = completion::Usage {
			input_tokens: response.usage.prompt_tokens as u64,
			output_tokens: response.usage.completion_tokens as u64,
			total_tokens: response.usage.total_tokens as u64,
			cached_input_tokens: 0,
		};

		Ok(completion::CompletionResponse {
			choice,
			usage,
			raw_response: response,
		})
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub(in crate::providers::huggingface) struct HuggingfaceCompletionRequest {
	model: String,
	pub messages: Vec<Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for HuggingfaceCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		let mut full_history: Vec<Message> = match &req.preamble {
			Some(preamble) => vec![Message::system(preamble)],
			None => vec![],
		};
		if let Some(docs) = req.normalized_documents() {
			let docs: Vec<Message> = docs.try_into()?;
			full_history.extend(docs);
		}

		let chat_history: Vec<Message> = req
			.chat_history
			.clone()
			.into_iter()
			.map(|message| message.try_into())
			.collect::<Result<Vec<Vec<Message>>, _>>()?
			.into_iter()
			.flatten()
			.collect();

		full_history.extend(chat_history);

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openai::completion::ToolChoice::try_from)
			.transpose()?;

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			tools: req
				.tools
				.clone()
				.into_iter()
				.map(ToolDefinition::from)
				.collect::<Vec<_>>(),
			tool_choice,
			additional_params: req.additional_params,
		})
	}
}
