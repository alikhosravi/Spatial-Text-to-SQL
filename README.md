Multi-Agent Framework for Spatial Text-to-SQL
Making Spatial Analysis Accessible Through Collaborative AI Agents

Overview

This repository accompanies the research paper:
> **“From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL”**


The project addresses a key challenge in spatial data analysis: the difficulty non-experts face in writing complex SQL and PostGIS queries. While Large Language Models (LLMs) can translate text to SQL, they often fall short on spatial reasoning and syntactic precision.
To overcome these issues, this work introduces a multi-agent system designed to collaboratively and accurately translate natural language questions into executable spatial SQL queries.

🧩 Abstract

The complexity of Structured Query Language (SQL) and the specialized nature of geospatial functions in tools like PostGIS present significant barriers to non-experts seeking to analyze spatial data. While Large Language Models (LLMs) offer promise for translating natural language into SQL (Text-to-SQL), single-agent approaches often struggle with the semantic and syntactic complexities of spatial queries.

To address this, we propose a novel multi-agent framework designed to accurately translate natural language questions into spatial SQL queries. Our framework integrates several innovative components:

📚 Knowledge base with programmatic schema profiling and semantic enrichment

⚙️ Collaborative multi-agent pipeline with specialized roles:

🤖 Entity extraction agent

🤖 Metadata retrieval agent

🤖 Query logic formulation agent

🤖 SQL generation agent

🤖 Review agent for programmatic and semantic validation

We evaluated the system using both the KaggleDBQA benchmark (non-spatial) and a new spatial benchmark we developed, which includes diverse geometry types, spatial predicates, and multiple query complexity levels.
