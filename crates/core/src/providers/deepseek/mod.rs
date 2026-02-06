//! DeepSeek API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::deepseek;
//!
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```

pub mod client;
pub mod completion;

pub use client::{Client, ClientBuilder, DeepSeek};
pub use completion::{
	CompletionModel, CompletionResponse, DEEPSEEK_CHAT, DEEPSEEK_REASONER,
	StreamingCompletionResponse,
};
