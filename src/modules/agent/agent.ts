import { Embeddings } from "@langchain/core/embeddings";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import initRephraseChain, { RephraseQuestionInput } from "./chains/rephrase-question.chain";
import { BaseChatModel } from "langchain/chat_models/base";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { getHistory } from "./history";
import initTools from "./tools";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";

export default async function initAgent(
  llm: BaseChatModel,
  embeddings: Embeddings,
  graph: Neo4jGraph
) {
  const tools = await initTools(llm, embeddings, graph);

  const prompt = await pull<ChatPromptTemplate>(
    "hwchase17/openai-functions-agent"
  );

  const agent = await createOpenAIFunctionsAgent({
    llm,
    tools,
    prompt
  });

  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: true
  });

  const rephraseQuestionChain = await initRephraseChain(llm);

  return RunnablePassthrough.assign<{ input: string; sessionId: string }, any>({
    history: async (_input, options) => {
      const sessionId = options?.configurable?.sessionId;
      if (!sessionId) {
        throw new Error("Session ID is missing in the configuration.");
      }
      const history = await getHistory(sessionId);
      return history;
    },
  })
    .assign({
      rephrasedQuestion: (input: RephraseQuestionInput, config: any) =>
        rephraseQuestionChain.invoke(input, config),
    })
    .pipe(executor)
    .pick("output");
}