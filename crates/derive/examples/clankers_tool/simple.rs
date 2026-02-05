use clankers::client::{CompletionClient, ProviderClient};
use clankers::completion::Prompt;
use clankers::providers;
use clankers_derive::clankers_tool;

// Simple example with no attributes (`required` is still needed for OpenAI's strict function tool calling)
#[clankers_tool(required(a, b))]
fn add(a: i32, b: i32) -> Result<i32, clankers::tool::ToolError> {
	Ok(a + b)
}

#[clankers_tool(required(a, b))]
fn subtract(a: i32, b: i32) -> Result<i32, clankers::tool::ToolError> {
	Ok(a - b)
}

#[clankers_tool(required(a, b))]
fn multiply(a: i32, b: i32) -> Result<i32, clankers::tool::ToolError> {
	Ok(a * b)
}

#[clankers_tool(required(a, b))]
fn divide(a: i32, b: i32) -> Result<i32, clankers::tool::ToolError> {
	if b == 0 {
		Err(clankers::tool::ToolError::ToolCallError(
			"Division by zero".into(),
		))
	} else {
		Ok(a / b)
	}
}

#[clankers_tool]
fn answer_secret_question() -> Result<(bool, bool, bool, bool, bool), clankers::tool::ToolError> {
	Ok((false, false, true, false, false))
}

#[clankers_tool]
fn how_many_rs(s: String) -> Result<usize, clankers::tool::ToolError> {
	Ok(s.chars()
		.filter(|c| *c == 'r' || *c == 'R')
		.collect::<Vec<_>>()
		.len())
}

#[clankers_tool]
fn sum_numbers(numbers: Vec<i64>) -> Result<i64, clankers::tool::ToolError> {
	Ok(numbers.iter().sum())
}

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt().pretty().init();

	let calculator_agent = providers::openai::Client::from_env()
		.agent(providers::openai::GPT_4O)
		.preamble("You are an agent with tools access, always use the tools")
		.max_tokens(1024)
		.tool(Add)
		.build();

	for prompt in [
		"What tools do you have?",
		"Calculate 5 + 3",
		"What is 10 + 20?",
		"Add 100 and 200",
	] {
		println!("User: {prompt}");
		println!("Agent: {}", calculator_agent.prompt(prompt).await.unwrap());
	}
}
