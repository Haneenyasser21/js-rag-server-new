import { config } from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import fetch from "node-fetch";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

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

function sanitizeMetadata(metadata) {
  const sanitized = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      sanitized[key] = value;
    } else if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
      sanitized[key] = value;
    } else {
      if (key === "loc" && typeof value === "object") {
        continue;
      }
      sanitized[key] = String(value);
    }
  }
  return sanitized;
}

export default async function handler(req, res) {
    // Allow requests from Wix (or all origins for testing)
    res.setHeader("Access-Control-Allow-Origin", "*"); // Change to specific Wix domain in production
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Allow-Credentials", "true");

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    if (req.method === "POST") {
        // Handle PDF ingestion
        const { pdfUrl } = req.body;
        if (!pdfUrl) {
            console.log("pdfUrl parameter is missing");
            return res.status(400).json({ error: "pdfUrl is required" });
        }

        try {
            // Fetch the PDF
            console.log(`Fetching PDF from ${pdfUrl}`);
            const response = await fetch(pdfUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch PDF: ${response.statusText}`);
            }
            const pdfBuffer = await response.buffer();

            // Load and split the PDF
            console.log("Loading PDF...");
            const loader = new PDFLoader(pdfBuffer);
            const docs = await loader.load();

            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
            });
            const chunks = await splitter.splitDocuments(docs);
            console.log(`Split into ${chunks.length} chunks`);

            // Embed and upsert to Pinecone
            const vectors = [];
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i];
                console.log(`Embedding chunk ${i + 1}/${chunks.length}`);
                const embedding = await embeddings.embedQuery(chunk.pageContent);
                const sanitizedMetadata = sanitizeMetadata(chunk.metadata || {});
                vectors.push({
                    id: `chunk-${Date.now()}-${i}`, // Unique ID for each chunk
                    values: embedding,
                    metadata: { text: chunk.pageContent, source: pdfUrl, ...sanitizedMetadata },
                });
            }

            console.log("Uploading vectors to Pinecone...");
            await index.upsert(vectors);
            console.log("Vectors uploaded successfully");

            res.status(200).json({ status: "PDF processed" });
        } catch (error) {
            console.error("Error in PDF ingestion:", error.message, error.stack);
            res.status(500).json({ error: error.message });
        }
    } else if (req.method === "GET") {
        // Handle query (existing logic)
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
    } else {
        res.status(405).json({ error: "Method not allowed" });
    }
}