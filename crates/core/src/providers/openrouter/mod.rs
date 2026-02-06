//! OpenRouter Inference API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::openrouter;
//!
//! let client = openrouter::Client::new("YOUR_API_KEY");
//!
//! let llama_3_1_8b = client.completion_model(openrouter::LLAMA_3_1_8B);
//! ```

pub mod client;
pub mod completion;
pub mod streaming;

pub use client::{Client, ClientBuilder};
pub use completion::{
	CLAUDE_3_7_SONNET, CompletionModel, GEMINI_FLASH_2_0, PERPLEXITY_SONAR_PRO, QWEN_QWQ_32B,
	ToolChoice,
};
