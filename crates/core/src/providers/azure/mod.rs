//! Azure OpenAI API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::azure;
//! use clankers::client::CompletionClient;
//!
//! let client: azure::Client<reqwest::Client> = azure::Client::builder()
//!     .api_key("test")
//!     .azure_endpoint("test".to_string()) // add your endpoint here!
//!     .build()?;
//!
//! let gpt4o = client.completion_model(azure::GPT_4O);
//! ```
//!
//! ## Authentication
//! The authentication type used for the `azure` module is [`AzureOpenAIAuth`].
//!
//! By default, using a type that implements `Into<String>` as the input for the client builder will turn the type into a bearer auth token.
//! If you want to use an API key, you need to use the type specifically.

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod embedding;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::AudioGenerationModel;
pub use client::{AzureOpenAIAuth, AzureOpenAIClientParams, Client, ClientBuilder};
pub use completion::CompletionModel;
pub use embedding::{EmbeddingModel, EmbeddingResponse};
#[cfg(feature = "image")]
pub use image_generation::ImageGenerationModel;
pub use transcription::TranscriptionModel;

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-mini` completion model
pub const O1_MINI: &str = "o1-mini";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-realtime-preview` completion model
pub const GPT_4O_REALTIME_PREVIEW: &str = "gpt-4o-realtime-preview";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
/// `gpt-3.5-turbo-16k` completion model
pub const GPT_35_TURBO_16K: &str = "gpt-3.5-turbo-16k";

#[cfg(test)]
mod azure_tests {
	use super::*;
	use crate::OneOrMany;
	use crate::client::ProviderClient;
	use crate::client::completion::CompletionClient;
	use crate::client::embeddings::EmbeddingsClient;
	use crate::completion::{CompletionModel, CompletionRequest};
	use crate::embeddings::EmbeddingModel;

	#[tokio::test]
	#[ignore]
	async fn test_azure_embedding() {
		let _ = tracing_subscriber::fmt::try_init();

		let client = Client::<reqwest::Client>::from_env();
		let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);
		let embeddings = model
			.embed_texts(vec!["Hello, world!".to_string()])
			.await
			.unwrap();

		tracing::info!("Azure embedding: {:?}", embeddings);
	}

	#[tokio::test]
	#[ignore]
	async fn test_azure_embedding_dimensions() {
		let _ = tracing_subscriber::fmt::try_init();

		let ndims = 256;
		let client = Client::<reqwest::Client>::from_env();
		let model = client.embedding_model_with_ndims(TEXT_EMBEDDING_3_SMALL, ndims);
		let embedding = model.embed_text("Hello, world!").await.unwrap();

		assert!(embedding.vec.len() == ndims);

		tracing::info!("Azure dimensions embedding: {:?}", embedding);
	}

	#[tokio::test]
	#[ignore]
	async fn test_azure_completion() {
		let _ = tracing_subscriber::fmt::try_init();

		let client = Client::<reqwest::Client>::from_env();
		let model = client.completion_model(GPT_4O_MINI);
		let completion = model
			.completion(CompletionRequest {
				preamble: Some("You are a helpful assistant.".to_string()),
				chat_history: OneOrMany::one("Hello!".into()),
				documents: vec![],
				max_tokens: Some(100),
				temperature: Some(0.0),
				tools: vec![],
				tool_choice: None,
				additional_params: None,
			})
			.await
			.unwrap();

		tracing::info!("Azure completion: {:?}", completion);
	}
}
