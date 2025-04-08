import { config } from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";

config();

console.log("Initializing Pinecone client...");
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
console.log("Pinecone client initialized");

console.log("Accessing Pinecone index 'js-rag-server'...");
const index = pinecone.Index("js-rag-server");
console.log("Pinecone index accessed");

console.log("Initializing OpenAI embeddings...");
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});
console.log("OpenAI embeddings initialized");

async function embedQueryWithTimeout(query, timeoutMs = 5000) {
  console.log(`Embedding query: "${query}" with timeout ${timeoutMs}ms`);
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Embedding query timed out")), timeoutMs);
  });
  return Promise.race([embeddings.embedQuery(query), timeout]);
}

export default async function handler(req, res) {
  console.log("Handler invoked with request:", req.method, req.query);

  if (req.method !== "GET") {
    console.log("Method not allowed:", req.method);
    return res.status(405).json({ error: "Method not allowed" });
  }

  const query = req.query.q;
  if (!query) {
    console.log("Query parameter 'q' is missing");
    return res.status(400).json({ error: "Query parameter 'q' is required" });
  }

  try {
    console.time("embedQuery");
    const queryEmbedding = await embedQueryWithTimeout(query, 5000);
    console.timeEnd("embedQuery");

    console.time("similaritySearch");
    const results = await index.query({
      vector: queryEmbedding,
      topK: 3,
      includeMetadata: true,
    });
    console.timeEnd("similaritySearch");

    console.log("Query results:", results.matches.length, "matches found");

    res.status(200).json(
      results.matches.map((match) => ({
        content: match.metadata.text || "No content available",
        metadata: match.metadata,
      }))
    );
  } catch (error) {
    console.error("Error in handler:", error.message, error.stack);
    res.status(500).json({ error: error.message });
  }
}