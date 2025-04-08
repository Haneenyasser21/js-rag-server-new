import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import { config } from "dotenv";

config();

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.Index("js-rag-server");

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Function to sanitize metadata for Pinecone
function sanitizeMetadata(metadata) {
  const sanitized = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      sanitized[key] = value;
    } else if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
      sanitized[key] = value;
    } else {
      // Convert unsupported types to strings or skip them
      if (key === "loc" && typeof value === "object") {
        // Skip the 'loc' field since it contains nested objects
        continue;
      }
      sanitized[key] = String(value);
    }
  }
  return sanitized;
}

async function main() {
  try {
    console.log("Loading documents...");
    const loader = new DirectoryLoader(
      path.join(process.cwd(), "data/books"),
      { ".md": (path) => new TextLoader(path) }
    );
    const docs = await loader.load();
    console.log(`Loaded ${docs.length} documents`);

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunks = await splitter.splitDocuments(docs);
    console.log(`Split into ${chunks.length} chunks`);

    const vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await embeddings.embedQuery(chunk.pageContent);
      const sanitizedMetadata = sanitizeMetadata(chunk.metadata);
      vectors.push({
        id: `chunk-${i}`,
        values: embedding,
        metadata: { text: chunk.pageContent, ...sanitizedMetadata },
      });
    }

    console.log("Uploading vectors to Pinecone...");
    await index.upsert(vectors);
    console.log("Vectors uploaded successfully");
  } catch (error) {
    console.error("Error:", error.message, error.stack);
  }
}

main();