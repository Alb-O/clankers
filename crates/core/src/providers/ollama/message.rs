use std::convert::TryFrom;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{DocumentSourceKind, ImageDetail, Text};
use crate::{OneOrMany, completion, json_utils, message};

// ---------- Tool Definition Conversion ----------

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
	#[serde(rename = "type")]
	pub type_field: String, // Fixed as "function"
	pub function: completion::ToolDefinition,
}

/// Convert internal ToolDefinition (from the completion module) into Ollama's tool definition.
impl From<crate::completion::ToolDefinition> for ToolDefinition {
	fn from(tool: crate::completion::ToolDefinition) -> Self {
		ToolDefinition {
			type_field: "function".to_owned(),
			function: completion::ToolDefinition {
				name: tool.name,
				description: tool.description,
				parameters: tool.parameters,
			},
		}
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
	#[serde(default, rename = "type")]
	pub r#type: ToolType,
	pub function: Function,
}
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
	#[default]
	Function,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
	pub name: String,
	pub arguments: Value,
}

// ---------- Provider Message Definition ----------

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
	User {
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		images: Option<Vec<String>>,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
	},
	Assistant {
		#[serde(default)]
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		thinking: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		images: Option<Vec<String>>,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
		#[serde(default, deserialize_with = "json_utils::null_or_vec")]
		tool_calls: Vec<ToolCall>,
	},
	System {
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		images: Option<Vec<String>>,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
	},
	#[serde(rename = "tool")]
	ToolResult {
		#[serde(rename = "tool_name")]
		name: String,
		content: String,
	},
}

/// -----------------------------
/// Provider Message Conversions
/// -----------------------------
/// Conversion from an internal Clankers message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
impl TryFrom<crate::message::Message> for Vec<Message> {
	type Error = crate::message::MessageError;
	fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
		use crate::message::Message as InternalMessage;
		match internal_msg {
			InternalMessage::User { content, .. } => {
				let (tool_results, other_content): (Vec<_>, Vec<_>) =
					content.into_iter().partition(|content| {
						matches!(content, crate::message::UserContent::ToolResult(_))
					});

				if !tool_results.is_empty() {
					tool_results
						.into_iter()
						.map(|content| match content {
							crate::message::UserContent::ToolResult(
								crate::message::ToolResult { id, content, .. },
							) => {
								// Ollama expects a single string for tool results, so we concatenate
								let content_string = content
									.into_iter()
									.map(|content| match content {
										crate::message::ToolResultContent::Text(text) => text.text,
										_ => "[Non-text content]".to_string(),
									})
									.collect::<Vec<_>>()
									.join("\n");

								Ok::<_, crate::message::MessageError>(Message::ToolResult {
									name: id,
									content: content_string,
								})
							}
							_ => unreachable!(),
						})
						.collect::<Result<Vec<_>, _>>()
				} else {
					// Ollama requires separate text content and images array
					let (texts, images) = other_content.into_iter().fold(
						(Vec::new(), Vec::new()),
						|(mut texts, mut images), content| {
							match content {
								crate::message::UserContent::Text(crate::message::Text {
									text,
								}) => texts.push(text),
								crate::message::UserContent::Image(crate::message::Image {
									data: DocumentSourceKind::Base64(data),
									..
								}) => images.push(data),
								crate::message::UserContent::Document(
									crate::message::Document {
										data:
											DocumentSourceKind::Base64(data)
											| DocumentSourceKind::String(data),
										..
									},
								) => texts.push(data),
								_ => {} // Audio not supported by Ollama
							}
							(texts, images)
						},
					);

					Ok(vec![Message::User {
						content: texts.join(" "),
						images: if images.is_empty() {
							None
						} else {
							Some(
								images
									.into_iter()
									.map(|x| x.to_string())
									.collect::<Vec<String>>(),
							)
						},
						name: None,
					}])
				}
			}
			InternalMessage::Assistant { content, .. } => {
				let mut thinking: Option<String> = None;
				let mut text_content = Vec::new();
				let mut tool_calls = Vec::new();

				for content in content.into_iter() {
					match content {
						crate::message::AssistantContent::Text(text) => {
							text_content.push(text.text)
						}
						crate::message::AssistantContent::ToolCall(tool_call) => {
							tool_calls.push(tool_call)
						}
						crate::message::AssistantContent::Reasoning(
							crate::message::Reasoning { reasoning, .. },
						) => {
							thinking = Some(reasoning.first().cloned().unwrap_or(String::new()));
						}
						crate::message::AssistantContent::Image(_) => {
							return Err(crate::message::MessageError::ConversionError(
								"Ollama currently doesn't support images.".into(),
							));
						}
					}
				}

				// `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
				//  so either `content` or `tool_calls` will have some content.
				Ok(vec![Message::Assistant {
					content: text_content.join(" "),
					thinking,
					images: None,
					name: None,
					tool_calls: tool_calls
						.into_iter()
						.map(|tool_call| tool_call.into())
						.collect::<Vec<_>>(),
				}])
			}
		}
	}
}

/// Conversion from provider Message to a completion message.
/// This is needed so that responses can be converted back into chat history.
impl From<Message> for crate::completion::Message {
	fn from(msg: Message) -> Self {
		match msg {
			Message::User { content, .. } => crate::completion::Message::User {
				content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
					text: content,
				})),
			},
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				let mut assistant_contents =
					vec![crate::completion::message::AssistantContent::Text(Text {
						text: content,
					})];
				for tc in tool_calls {
					assistant_contents.push(
						crate::completion::message::AssistantContent::tool_call(
							tc.function.name.clone(),
							tc.function.name,
							tc.function.arguments,
						),
					);
				}
				crate::completion::Message::Assistant {
					id: None,
					content: OneOrMany::many(assistant_contents).unwrap(),
				}
			}
			// System and ToolResult are converted to User message as needed.
			Message::System { content, .. } => crate::completion::Message::User {
				content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
					text: content,
				})),
			},
			Message::ToolResult { name, content } => crate::completion::Message::User {
				content: OneOrMany::one(message::UserContent::tool_result(
					name,
					OneOrMany::one(message::ToolResultContent::text(content)),
				)),
			},
		}
	}
}

impl Message {
	/// Constructs a system message.
	pub fn system(content: &str) -> Self {
		Message::System {
			content: content.to_owned(),
			images: None,
			name: None,
		}
	}
}

// ---------- Additional Message Types ----------

impl From<crate::message::ToolCall> for ToolCall {
	fn from(tool_call: crate::message::ToolCall) -> Self {
		Self {
			r#type: ToolType::Function,
			function: Function {
				name: tool_call.function.name,
				arguments: tool_call.function.arguments,
			},
		}
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
	#[serde(default)]
	r#type: SystemContentType,
	text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
	#[default]
	Text,
}

impl From<String> for SystemContent {
	fn from(s: String) -> Self {
		SystemContent {
			r#type: SystemContentType::default(),
			text: s,
		}
	}
}

impl FromStr for SystemContent {
	type Err = std::convert::Infallible;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(SystemContent {
			r#type: SystemContentType::default(),
			text: s.to_string(),
		})
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AssistantContent {
	pub text: String,
}

impl FromStr for AssistantContent {
	type Err = std::convert::Infallible;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(AssistantContent { text: s.to_owned() })
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
	Text { text: String },
	Image { image_url: ImageUrl },
	// Audio variant removed as Ollama API does not support audio input.
}

impl FromStr for UserContent {
	type Err = std::convert::Infallible;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(UserContent::Text { text: s.to_owned() })
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
	pub url: String,
	#[serde(default)]
	pub detail: ImageDetail,
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	// Test conversion from provider Message to completion Message.
	#[test]
	fn test_message_conversion() {
		// Construct a provider Message (User variant with String content).
		let provider_msg = Message::User {
			content: "Test message".to_owned(),
			images: None,
			name: None,
		};
		// Convert it into a completion::Message.
		let comp_msg: crate::completion::Message = provider_msg.into();
		match comp_msg {
			crate::completion::Message::User { content } => {
				// Assume OneOrMany<T> has a method first() to access the first element.
				let first_content = content.first();
				// The expected type is crate::completion::message::UserContent::Text wrapping a Text struct.
				match first_content {
					crate::completion::message::UserContent::Text(text_struct) => {
						assert_eq!(text_struct.text, "Test message");
					}
					_ => panic!("Expected text content in conversion"),
				}
			}
			_ => panic!("Conversion from provider Message to completion Message failed"),
		}
	}

	// Test conversion of internal tool definition to Ollama's ToolDefinition format.
	#[test]
	fn test_tool_definition_conversion() {
		// Internal tool definition from the completion module.
		let internal_tool = crate::completion::ToolDefinition {
			name: "get_current_weather".to_owned(),
			description: "Get the current weather for a location".to_owned(),
			parameters: json!({
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The location to get the weather for, e.g. San Francisco, CA"
					},
					"format": {
						"type": "string",
						"description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
						"enum": ["celsius", "fahrenheit"]
					}
				},
				"required": ["location", "format"]
			}),
		};
		// Convert internal tool to Ollama's tool definition.
		let ollama_tool: ToolDefinition = internal_tool.into();
		assert_eq!(ollama_tool.type_field, "function");
		assert_eq!(ollama_tool.function.name, "get_current_weather");
		assert_eq!(
			ollama_tool.function.description,
			"Get the current weather for a location"
		);
		// Check JSON fields in parameters.
		let params = &ollama_tool.function.parameters;
		assert_eq!(params["properties"]["location"]["type"], "string");
	}

	// Test message conversion with thinking content
	#[test]
	fn test_message_conversion_with_thinking() {
		// Create an internal message with reasoning content
		let reasoning_content = crate::message::Reasoning {
			id: None,
			reasoning: vec!["Step 1: Consider the problem".to_string()],
			signature: None,
		};

		let internal_msg = crate::message::Message::Assistant {
			id: None,
			content: crate::OneOrMany::many(vec![
				crate::message::AssistantContent::Reasoning(reasoning_content),
				crate::message::AssistantContent::Text(crate::message::Text {
					text: "The answer is X".to_string(),
				}),
			])
			.unwrap(),
		};

		// Convert to provider Message
		let provider_msgs: Vec<Message> = internal_msg.try_into().unwrap();
		assert_eq!(provider_msgs.len(), 1);

		if let Message::Assistant {
			thinking, content, ..
		} = &provider_msgs[0]
		{
			assert_eq!(thinking.as_ref().unwrap(), "Step 1: Consider the problem");
			assert_eq!(content, "The answer is X");
		} else {
			panic!("Expected Assistant message with thinking");
		}
	}
}
