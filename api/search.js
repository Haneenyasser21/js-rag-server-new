import { config } from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";

config();

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.Index("js-rag-server");

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

async function embedQueryWithTimeout(query, timeoutMs = 3000) {
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Embedding query timed out")), timeoutMs);
  });
  return Promise.race([embeddings.embedQuery(query), timeout]);
}

export default async function handler(req, res) {
  // Ensure the request method is GET
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // Get the query parameter 'q'
  const query = req.query.q;
  if (!query) {
    return res.status(400).json({ error: "Query parameter 'q' is required" });
  }

  try {
    // Embed the query
    console.time("embedQuery");
    const queryEmbedding = await embedQueryWithTimeout(query, 3000);
    console.timeEnd("embedQuery");

    // Query Pinecone for similar vectors
    console.time("similaritySearch");
    const results = await index.query({
      vector: queryEmbedding,
      topK: 3,
      includeMetadata: true,
    });
    console.timeEnd("similaritySearch");

    // Format and return the results
    res.status(200).json(
      results.matches.map((match) => ({
        content: match.metadata.text || "No content available",
        metadata: match.metadata,
      }))
    );
  } catch (error) {
    console.error("Error in handler:", error.message);
    res.status(500).json({ error: error.message });
  }
}
