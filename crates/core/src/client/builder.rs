use crate::completion::CompletionError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Provider '{0}' not found")]
	NotFound(String),
	#[error("Provider '{provider}' cannot be coerced to a '{role}'")]
	NotCapable { provider: String, role: String },
	#[error("Error generating response\n{0}")]
	Completion(#[from] CompletionError),
}
