use clankers::completion::Prompt;
use clankers::prelude::*;
use clankers::providers::gemini::completion::GEMINI_2_5_PRO_EXP_03_25;
use clankers::providers::openrouter;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
	// Create OpenAI client
	let client = openrouter::Client::from_env();

	// Create agent with a single context prompt
	let comedian_agent = client
		.agent(GEMINI_2_5_PRO_EXP_03_25)
		.preamble("You are a comedian here to entertain the user using humour and jokes.")
		.build();

	// Prompt the agent and print the response
	let response = comedian_agent.prompt("Entertain me!").await?;

	println!("{response}");

	Ok(())
}
