use std::convert::Infallible;
use std::str::FromStr;

use super::message::{
	AssistantContent, Audio, AudioMediaType, Document, DocumentMediaType, DocumentSourceKind,
	Image, ImageDetail, ImageMediaType, MediaType, Message, MimeType, Reasoning, Text, ToolCall,
	ToolFunction, ToolResult, ToolResultContent, UserContent, VideoMediaType,
};
use crate::OneOrMany;

impl Message {
	/// This helper method is primarily used to extract the first string prompt from a `Message`.
	/// Since `Message` might have more than just text content, we need to find the first text.
	pub(crate) fn rag_text(&self) -> Option<String> {
		match self {
			Message::User { content } => {
				for item in content.iter() {
					if let UserContent::Text(Text { text }) = item {
						return Some(text.clone());
					}
				}
				None
			}
			_ => None,
		}
	}

	/// Helper constructor to make creating user messages easier.
	pub fn user(text: impl Into<String>) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::text(text)),
		}
	}

	/// Helper constructor to make creating assistant messages easier.
	pub fn assistant(text: impl Into<String>) -> Self {
		Message::Assistant {
			id: None,
			content: OneOrMany::one(AssistantContent::text(text)),
		}
	}

	/// Helper constructor to make creating assistant messages easier.
	pub fn assistant_with_id(id: String, text: impl Into<String>) -> Self {
		Message::Assistant {
			id: Some(id),
			content: OneOrMany::one(AssistantContent::text(text)),
		}
	}

	/// Helper constructor to make creating tool result messages easier.
	pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::ToolResult(ToolResult {
				id: id.into(),
				call_id: None,
				content: OneOrMany::one(ToolResultContent::text(content)),
			})),
		}
	}

	pub fn tool_result_with_call_id(
		id: impl Into<String>,
		call_id: Option<String>,
		content: impl Into<String>,
	) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::ToolResult(ToolResult {
				id: id.into(),
				call_id,
				content: OneOrMany::one(ToolResultContent::text(content)),
			})),
		}
	}
}

impl UserContent {
	/// Helper constructor to make creating user text content easier.
	pub fn text(text: impl Into<String>) -> Self {
		UserContent::Text(text.into().into())
	}

	/// Helper constructor to make creating user image content easier.
	pub fn image_base64(
		data: impl Into<String>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		UserContent::Image(Image {
			data: DocumentSourceKind::Base64(data.into()),
			media_type,
			detail,
			additional_params: None,
		})
	}

	/// Helper constructor to make creating user image content from raw unencoded bytes easier.
	pub fn image_raw(
		data: impl Into<Vec<u8>>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		UserContent::Image(Image {
			data: DocumentSourceKind::Raw(data.into()),
			media_type,
			detail,
			..Default::default()
		})
	}

	/// Helper constructor to make creating user image content easier.
	pub fn image_url(
		url: impl Into<String>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		UserContent::Image(Image {
			data: DocumentSourceKind::Url(url.into()),
			media_type,
			detail,
			additional_params: None,
		})
	}

	/// Helper constructor to make creating user audio content easier.
	pub fn audio(data: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
		UserContent::Audio(Audio {
			data: DocumentSourceKind::Base64(data.into()),
			media_type,
			additional_params: None,
		})
	}

	/// Helper constructor to make creating user audio content from raw unencoded bytes easier.
	pub fn audio_raw(data: impl Into<Vec<u8>>, media_type: Option<AudioMediaType>) -> Self {
		UserContent::Audio(Audio {
			data: DocumentSourceKind::Raw(data.into()),
			media_type,
			..Default::default()
		})
	}

	/// Helper to create an audio resource from a URL
	pub fn audio_url(url: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
		UserContent::Audio(Audio {
			data: DocumentSourceKind::Url(url.into()),
			media_type,
			..Default::default()
		})
	}

	/// Helper constructor to make creating user document content easier.
	/// This creates a document that assumes the data being passed in is a raw string.
	pub fn document(data: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
		let data: String = data.into();
		UserContent::Document(Document {
			data: DocumentSourceKind::string(&data),
			media_type,
			additional_params: None,
		})
	}

	/// Helper to create a document from raw unencoded bytes
	pub fn document_raw(data: impl Into<Vec<u8>>, media_type: Option<DocumentMediaType>) -> Self {
		UserContent::Document(Document {
			data: DocumentSourceKind::Raw(data.into()),
			media_type,
			..Default::default()
		})
	}

	/// Helper to create a document from a URL
	pub fn document_url(url: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
		UserContent::Document(Document {
			data: DocumentSourceKind::Url(url.into()),
			media_type,
			..Default::default()
		})
	}

	/// Helper constructor to make creating user tool result content easier.
	pub fn tool_result(id: impl Into<String>, content: OneOrMany<ToolResultContent>) -> Self {
		UserContent::ToolResult(ToolResult {
			id: id.into(),
			call_id: None,
			content,
		})
	}

	/// Helper constructor to make creating user tool result content easier.
	pub fn tool_result_with_call_id(
		id: impl Into<String>,
		call_id: String,
		content: OneOrMany<ToolResultContent>,
	) -> Self {
		UserContent::ToolResult(ToolResult {
			id: id.into(),
			call_id: Some(call_id),
			content,
		})
	}
}

impl AssistantContent {
	/// Helper constructor to make creating assistant text content easier.
	pub fn text(text: impl Into<String>) -> Self {
		AssistantContent::Text(text.into().into())
	}

	/// Helper constructor to make creating assistant image content easier.
	pub fn image_base64(
		data: impl Into<String>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		AssistantContent::Image(Image {
			data: DocumentSourceKind::Base64(data.into()),
			media_type,
			detail,
			additional_params: None,
		})
	}

	/// Helper constructor to make creating assistant tool call content easier.
	pub fn tool_call(
		id: impl Into<String>,
		name: impl Into<String>,
		arguments: serde_json::Value,
	) -> Self {
		AssistantContent::ToolCall(ToolCall::new(
			id.into(),
			ToolFunction {
				name: name.into(),
				arguments,
			},
		))
	}

	pub fn tool_call_with_call_id(
		id: impl Into<String>,
		call_id: String,
		name: impl Into<String>,
		arguments: serde_json::Value,
	) -> Self {
		AssistantContent::ToolCall(
			ToolCall::new(
				id.into(),
				ToolFunction {
					name: name.into(),
					arguments,
				},
			)
			.with_call_id(call_id),
		)
	}

	pub fn reasoning(reasoning: impl AsRef<str>) -> Self {
		AssistantContent::Reasoning(Reasoning::new(reasoning.as_ref()))
	}
}

impl ToolResultContent {
	/// Helper constructor to make creating tool result text content easier.
	pub fn text(text: impl Into<String>) -> Self {
		ToolResultContent::Text(text.into().into())
	}

	/// Helper constructor to make tool result images from a base64-encoded string.
	pub fn image_base64(
		data: impl Into<String>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		ToolResultContent::Image(Image {
			data: DocumentSourceKind::Base64(data.into()),
			media_type,
			detail,
			additional_params: None,
		})
	}

	/// Helper constructor to make tool result images from a base64-encoded string.
	pub fn image_raw(
		data: impl Into<Vec<u8>>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		ToolResultContent::Image(Image {
			data: DocumentSourceKind::Raw(data.into()),
			media_type,
			detail,
			..Default::default()
		})
	}

	/// Helper constructor to make tool result images from a URL.
	pub fn image_url(
		url: impl Into<String>,
		media_type: Option<ImageMediaType>,
		detail: Option<ImageDetail>,
	) -> Self {
		ToolResultContent::Image(Image {
			data: DocumentSourceKind::Url(url.into()),
			media_type,
			detail,
			additional_params: None,
		})
	}

	/// Parse a tool output string into appropriate ToolResultContent(s).
	///
	/// Supports three formats:
	/// 1. Simple text: Any string → `OneOrMany::one(Text)`
	/// 2. Image JSON: `{"type": "image", "data": "...", "mimeType": "..."}` → `OneOrMany::one(Image)`
	/// 3. Hybrid JSON: `{"response": {...}, "parts": [...]}` → `OneOrMany::many([Text, Image, ...])`
	///
	/// If JSON parsing fails, treats the entire string as text.
	pub fn from_tool_output(output: impl Into<String>) -> OneOrMany<ToolResultContent> {
		let output_str = output.into();

		if let Ok(json) = serde_json::from_str::<serde_json::Value>(&output_str) {
			if json.get("response").is_some() || json.get("parts").is_some() {
				let mut results: Vec<ToolResultContent> = Vec::new();

				if let Some(response) = json.get("response") {
					results.push(ToolResultContent::Text(Text {
						text: response.to_string(),
					}));
				}

				if let Some(parts) = json.get("parts").and_then(|p| p.as_array()) {
					for part in parts {
						let is_image = part
							.get("type")
							.and_then(|t| t.as_str())
							.is_some_and(|t| t == "image");

						if !is_image {
							continue;
						}

						if let (Some(data), Some(mime_type)) = (
							part.get("data").and_then(|v| v.as_str()),
							part.get("mimeType").and_then(|v| v.as_str()),
						) {
							let data_kind =
								if data.starts_with("http://") || data.starts_with("https://") {
									DocumentSourceKind::Url(data.to_string())
								} else {
									DocumentSourceKind::Base64(data.to_string())
								};

							results.push(ToolResultContent::Image(Image {
								data: data_kind,
								media_type: ImageMediaType::from_mime_type(mime_type),
								detail: None,
								additional_params: None,
							}));
						}
					}
				}

				if !results.is_empty() {
					return OneOrMany::many(results).unwrap_or_else(|_| {
						OneOrMany::one(ToolResultContent::Text(output_str.into()))
					});
				}
			}

			let is_image = json
				.get("type")
				.and_then(|v| v.as_str())
				.is_some_and(|t| t == "image");

			if is_image
				&& let (Some(data), Some(mime_type)) = (
					json.get("data").and_then(|v| v.as_str()),
					json.get("mimeType").and_then(|v| v.as_str()),
				) {
				let data_kind = if data.starts_with("http://") || data.starts_with("https://") {
					DocumentSourceKind::Url(data.to_string())
				} else {
					DocumentSourceKind::Base64(data.to_string())
				};

				return OneOrMany::one(ToolResultContent::Image(Image {
					data: data_kind,
					media_type: ImageMediaType::from_mime_type(mime_type),
					detail: None,
					additional_params: None,
				}));
			}
		}

		OneOrMany::one(ToolResultContent::Text(output_str.into()))
	}
}

impl MimeType for MediaType {
	fn from_mime_type(mime_type: &str) -> Option<Self> {
		ImageMediaType::from_mime_type(mime_type)
			.map(MediaType::Image)
			.or_else(|| {
				DocumentMediaType::from_mime_type(mime_type)
					.map(MediaType::Document)
					.or_else(|| {
						AudioMediaType::from_mime_type(mime_type)
							.map(MediaType::Audio)
							.or_else(|| {
								VideoMediaType::from_mime_type(mime_type).map(MediaType::Video)
							})
					})
			})
	}

	fn to_mime_type(&self) -> &'static str {
		match self {
			MediaType::Image(media_type) => media_type.to_mime_type(),
			MediaType::Audio(media_type) => media_type.to_mime_type(),
			MediaType::Document(media_type) => media_type.to_mime_type(),
			MediaType::Video(media_type) => media_type.to_mime_type(),
		}
	}
}

impl MimeType for ImageMediaType {
	fn from_mime_type(mime_type: &str) -> Option<Self> {
		match mime_type {
			"image/jpeg" => Some(ImageMediaType::JPEG),
			"image/png" => Some(ImageMediaType::PNG),
			"image/gif" => Some(ImageMediaType::GIF),
			"image/webp" => Some(ImageMediaType::WEBP),
			"image/heic" => Some(ImageMediaType::HEIC),
			"image/heif" => Some(ImageMediaType::HEIF),
			"image/svg+xml" => Some(ImageMediaType::SVG),
			_ => None,
		}
	}

	fn to_mime_type(&self) -> &'static str {
		match self {
			ImageMediaType::JPEG => "image/jpeg",
			ImageMediaType::PNG => "image/png",
			ImageMediaType::GIF => "image/gif",
			ImageMediaType::WEBP => "image/webp",
			ImageMediaType::HEIC => "image/heic",
			ImageMediaType::HEIF => "image/heif",
			ImageMediaType::SVG => "image/svg+xml",
		}
	}
}

impl MimeType for DocumentMediaType {
	fn from_mime_type(mime_type: &str) -> Option<Self> {
		match mime_type {
			"application/pdf" => Some(DocumentMediaType::PDF),
			"text/plain" => Some(DocumentMediaType::TXT),
			"text/rtf" => Some(DocumentMediaType::RTF),
			"text/html" => Some(DocumentMediaType::HTML),
			"text/css" => Some(DocumentMediaType::CSS),
			"text/md" | "text/markdown" => Some(DocumentMediaType::MARKDOWN),
			"text/csv" => Some(DocumentMediaType::CSV),
			"text/xml" => Some(DocumentMediaType::XML),
			"application/x-javascript" | "text/x-javascript" => Some(DocumentMediaType::Javascript),
			"application/x-python" | "text/x-python" => Some(DocumentMediaType::Python),
			_ => None,
		}
	}

	fn to_mime_type(&self) -> &'static str {
		match self {
			DocumentMediaType::PDF => "application/pdf",
			DocumentMediaType::TXT => "text/plain",
			DocumentMediaType::RTF => "text/rtf",
			DocumentMediaType::HTML => "text/html",
			DocumentMediaType::CSS => "text/css",
			DocumentMediaType::MARKDOWN => "text/markdown",
			DocumentMediaType::CSV => "text/csv",
			DocumentMediaType::XML => "text/xml",
			DocumentMediaType::Javascript => "application/x-javascript",
			DocumentMediaType::Python => "application/x-python",
		}
	}
}

impl MimeType for AudioMediaType {
	fn from_mime_type(mime_type: &str) -> Option<Self> {
		match mime_type {
			"audio/wav" => Some(AudioMediaType::WAV),
			"audio/mp3" => Some(AudioMediaType::MP3),
			"audio/aiff" => Some(AudioMediaType::AIFF),
			"audio/aac" => Some(AudioMediaType::AAC),
			"audio/ogg" => Some(AudioMediaType::OGG),
			"audio/flac" => Some(AudioMediaType::FLAC),
			_ => None,
		}
	}

	fn to_mime_type(&self) -> &'static str {
		match self {
			AudioMediaType::WAV => "audio/wav",
			AudioMediaType::MP3 => "audio/mp3",
			AudioMediaType::AIFF => "audio/aiff",
			AudioMediaType::AAC => "audio/aac",
			AudioMediaType::OGG => "audio/ogg",
			AudioMediaType::FLAC => "audio/flac",
		}
	}
}

impl MimeType for VideoMediaType {
	fn from_mime_type(mime_type: &str) -> Option<Self>
	where
		Self: Sized,
	{
		match mime_type {
			"video/avi" => Some(VideoMediaType::AVI),
			"video/mp4" => Some(VideoMediaType::MP4),
			"video/mpeg" => Some(VideoMediaType::MPEG),
			&_ => None,
		}
	}

	fn to_mime_type(&self) -> &'static str {
		match self {
			VideoMediaType::AVI => "video/avi",
			VideoMediaType::MP4 => "video/mp4",
			VideoMediaType::MPEG => "video/mpeg",
		}
	}
}

impl std::str::FromStr for ImageDetail {
	type Err = ();

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_lowercase().as_str() {
			"low" => Ok(ImageDetail::Low),
			"high" => Ok(ImageDetail::High),
			"auto" => Ok(ImageDetail::Auto),
			_ => Err(()),
		}
	}
}

impl From<String> for Text {
	fn from(text: String) -> Self {
		Text { text }
	}
}

impl From<&String> for Text {
	fn from(text: &String) -> Self {
		text.to_owned().into()
	}
}

impl From<&str> for Text {
	fn from(text: &str) -> Self {
		text.to_owned().into()
	}
}

impl FromStr for Text {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(s.into())
	}
}

impl From<String> for Message {
	fn from(text: String) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Text(text.into())),
		}
	}
}

impl From<&str> for Message {
	fn from(text: &str) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Text(text.into())),
		}
	}
}

impl From<&String> for Message {
	fn from(text: &String) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Text(text.into())),
		}
	}
}

impl From<Text> for Message {
	fn from(text: Text) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Text(text)),
		}
	}
}

impl From<Image> for Message {
	fn from(image: Image) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Image(image)),
		}
	}
}

impl From<Audio> for Message {
	fn from(audio: Audio) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Audio(audio)),
		}
	}
}

impl From<Document> for Message {
	fn from(document: Document) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::Document(document)),
		}
	}
}

impl From<String> for ToolResultContent {
	fn from(text: String) -> Self {
		ToolResultContent::text(text)
	}
}

impl From<String> for AssistantContent {
	fn from(text: String) -> Self {
		AssistantContent::text(text)
	}
}

impl From<String> for UserContent {
	fn from(text: String) -> Self {
		UserContent::text(text)
	}
}

impl From<AssistantContent> for Message {
	fn from(content: AssistantContent) -> Self {
		Message::Assistant {
			id: None,
			content: OneOrMany::one(content),
		}
	}
}

impl From<UserContent> for Message {
	fn from(content: UserContent) -> Self {
		Message::User {
			content: OneOrMany::one(content),
		}
	}
}

impl From<OneOrMany<AssistantContent>> for Message {
	fn from(content: OneOrMany<AssistantContent>) -> Self {
		Message::Assistant { id: None, content }
	}
}

impl From<OneOrMany<UserContent>> for Message {
	fn from(content: OneOrMany<UserContent>) -> Self {
		Message::User { content }
	}
}

impl From<ToolCall> for Message {
	fn from(tool_call: ToolCall) -> Self {
		Message::Assistant {
			id: None,
			content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
		}
	}
}

impl From<ToolResult> for Message {
	fn from(tool_result: ToolResult) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::ToolResult(tool_result)),
		}
	}
}

impl From<ToolResultContent> for Message {
	fn from(tool_result_content: ToolResultContent) -> Self {
		Message::User {
			content: OneOrMany::one(UserContent::ToolResult(ToolResult {
				id: String::new(),
				call_id: None,
				content: OneOrMany::one(tool_result_content),
			})),
		}
	}
}
