import { write } from "../modules/graph"; // Required to run Cypher queries

/**
 * Stores user feedback for a chatbot response.
 * Adds either :HelpfulResponse or :UnhelpfulResponse label to the Response node.
 *
 * @param responseId - The Neo4j Response node ID
 * @param helpful - Boolean indicating whether the user found the response helpful
 */
export async function provideFeedback(responseId: string, helpful: boolean): Promise<void> {
  await write(
    `
    MATCH (r:Response {id: $responseId})
    CALL {
      WITH r
      WHERE $helpful = true
      SET r:HelpfulResponse
    }
    CALL {
      WITH r
      WHERE $helpful = false
      SET r:UnhelpfulResponse
    }
    `,
    { responseId, helpful }
  );
}