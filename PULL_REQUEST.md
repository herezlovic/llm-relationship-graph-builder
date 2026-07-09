# Pull Request: Simplify and Demystify Repository Documentation

## Overview
This pull request refactors the main [README.md](README.md) to explain the project's features and goals in plain, non-technical English. It removes all installation steps, deployment details, and framework dependencies, keeping the focus strictly on what the system does.

---

## Why Was This Change Done?
The previous documentation was heavily developer-oriented, detailing local and cloud setup steps, Docker configurations, and environment variables. While necessary for deployment, it overshadowed the application's purpose. 

This change was made to:
1. **Improve Accessibility:** Make the repository immediately understandable to non-technical users, product managers, or stakeholders.
2. **Focus on Functionality:** Provide a clean, high-level summary of the tool's core capabilities (data ingestion, AI mapping, interactive exploration, conversational Q&A, and usage tracking).
3. **Standardize Presentation:** Deliver a clean, professional, and visually structured homepage for the project.

---

## Summary of Changes

### 1. Documentation Cleanup
* Removed all technical prerequisites and setup steps (Python, Neo4j, Docker-compose).
* Removed backend and frontend deployment instructions (local run, cloud deployment, and Ollama configuration).
* Removed the environment variables reference table (`BACKEND ENV` and `FRONTEND ENV`).

### 2. Feature-Centric Rewrite
Rewrote the project documentation from scratch to highlight the following features in simple language:
* **Source Integration:** Explaining how the application pulls text from local files, websites, online videos, and cloud storage.
* **Relationship Mapping:** Describing how AI reads unstructured text to find key items and links them together.
* **Custom Modeling:** Outlining how users can direct the AI to focus on specific topics and connections.
* **Visual Map View:** Detailing how users can explore documents as connected circles and lines.
* **Conversational Assistant:** Highlighting the ability to ask natural-language questions and receive answers complete with source citations.
* **Usage Limits:** Highlighting the system's ability to track daily and monthly usage limits.
