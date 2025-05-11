// tag::prompt[]
import { PromptTemplate } from "@langchain/core/prompts";
import 'dotenv/config'; // Load environment variables from .env.local

const prompt = PromptTemplate.fromTemplate(`
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
`);
// end::prompt[]

// tag::llm[]
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY
});
// end::llm[]

// tag::parser[]
import { StringOutputParser } from "@langchain/core/output_parsers";

const parser = new StringOutputParser();
// end::parser[]
// tag::passthrough[]
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
// end::passthrough[]

/**
 *
 * To ensure type safety, you can define the input and output types

tag::types[]
RunnableSequence.from<InputType, OutputType>
end::types[]

* If you would like the `invoke()` function to accept a string, you can convert
* the input into an Object using a RunnablePassthrough.

// tag::passthrough[]
{
    fruit: new RunnablePassthrough(),
},
// end::passthrough[]
*/

// tag::chain[]
const chain = RunnableSequence.from([prompt, llm, parser]);
// end::chain[]

const main = async () => {
  const { performance } = await import("node:perf_hooks");

  const start = performance.now();

  const response = await chain.invoke({ fruit: "pineapple" });

  const end = performance.now();

  console.log(response);
  console.log(`⏱️ Response generated in ${(end - start).toFixed(2)} ms`);
};

main();
