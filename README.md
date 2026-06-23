Multi-Agent Framework for Spatial Text-to-SQL
Making Spatial Analysis Accessible Through Collaborative AI Agents

Overview

This repository accompanies the research paper:
> **“From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL”**


The project addresses a key challenge in spatial data analysis: the difficulty non-experts face in writing complex SQL and PostGIS queries. While Large Language Models (LLMs) can translate text to SQL, they often fall short on spatial reasoning and syntactic precision.
To overcome these issues, this work introduces a multi-agent system designed to collaboratively and accurately translate natural language questions into executable spatial SQL queries.

🧩 Abstract

The complexity of SQL and the spatial semantics of PostGIS create barriers for non-experts working with spatial data. Although large language models can translate natural language into SQL, spatial Text-to-SQL is more error-prone than general Text-to-SQL because it must resolve geographic intent, schema ambiguity, geometry-bearing tables and columns, spatial function choice, and coordinate reference system and measurement assumptions. We introduce a multi-agent framework that addresses these coupled challenges through staged interpretation, schema grounding, logical planning, SQL generation, and execution-based review. The framework is supported by a knowledge base with programmatic schema profiling, semantic enrichment, and embedding-based retrieval. We evaluated the framework on the non-spatial KaggleDBQA benchmark and on SpatialQueryQA, a new multi-level and coverage-oriented benchmark with diverse geometry types, workload categories, and spatial operations. On KaggleDBQA, the system reached 81.2% accuracy, 221 of 272 questions, after reviewer corrections. On SpatialQueryQA, the system achieved 87.7% accuracy, 79 of 90, compared with 76.7% without the review stage. These results show that decomposing the task into specialized but tightly coupled agents improves robustness, especially for spatially sensitive queries. The study improves access to spatial analysis and provides a practical step toward more reliable spatial Text-to-SQL systems and autonomous GIS.

We propose a novel multi-agent framework designed to accurately translate natural language questions into spatial SQL queries. Our framework integrates several innovative components:

📚 Knowledge base with programmatic schema profiling and semantic enrichment

⚙️ Collaborative multi-agent pipeline with specialized roles:

🤖 Entity extraction agent

🤖 Metadata retrieval agent

🤖 Query logic formulation agent

🤖 SQL generation agent

🤖 Review agent for programmatic and semantic validation

We evaluated the system using both the KaggleDBQA benchmark (non-spatial) and a new spatial benchmark we developed, which includes diverse geometry types, spatial predicates, and multiple query complexity levels.
