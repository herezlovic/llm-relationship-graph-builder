import logging
from graphdatascience import GraphDataScience
from src.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from src.shared.common_fn import get_value_from_env
from src.shared.common_fn import load_embedding_model,track_token_usage
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException

COMMUNITY_PROJECTION_NAME = "communities"
NODE_PROJECTION = "!Chunk&!Document&!__Community__&!__Claim__&!__RaptorNode__"
NODE_PROJECTION_ENTITY = "__Entity__"
MAX_WORKERS = 10
MAX_COMMUNITY_LEVELS = 3 
MIN_COMMUNITY_SIZE = 1 
COMMUNITY_CREATION_DEFAULT_MODEL = "openai_gpt_5_mini"


CREATE_COMMUNITY_GRAPH_PROJECTION = """
MATCH (source:{node_projection})-[]->(target:{node_projection})
WITH source, target, count(*) as weight
WITH gds.graph.project(
               '{project_name}',
               source,
               target,
               {{
               relationshipProperties: {{ weight: weight }}
               }},
               {{undirectedRelationshipTypes: ['*']}}
               ) AS g
RETURN
  g.graphName AS graph_name, g.nodeCount AS nodes, g.relationshipCount AS rels
"""

CREATE_COMMUNITY_CONSTRAINT = "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
CREATE_COMMUNITY_LEVELS = """
MATCH (e:`__Entity__`)
WHERE e.communities is NOT NULL
UNWIND range(0, size(e.communities) - 1 , 1) AS index
CALL {
  WITH e, index
  WITH e, index
  WHERE index = 0
  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET c.level = index
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
CALL {
  WITH e, index
  WITH e, index
  WHERE index > 0
  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET current.level = index
  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
  ON CREATE SET previous.level = index - 1
  MERGE (previous)-[:PARENT_COMMUNITY]->(current)
  RETURN count(*) AS count_1
}
RETURN count(*)
"""
CREATE_COMMUNITY_RANKS = """
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:!Chunk&!Document&!__Community__&!__Claim__&!__RaptorNode__)<-[HAS_ENTITY]-(:Chunk)<-[]-(d:Document)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
"""

CREATE_PARENT_COMMUNITY_RANKS = """
MATCH (c:__Community__)<-[:PARENT_COMMUNITY*]-(:__Community__)<-[:IN_COMMUNITY*]-(:!Chunk&!Document&!__Community__&!__Claim__&!__RaptorNode__)<-[HAS_ENTITY]-(:Chunk)<-[]-(d:Document)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
"""

CREATE_COMMUNITY_WEIGHTS = """
MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount
"""
CREATE_PARENT_COMMUNITY_WEIGHTS = """
MATCH (n:`__Community__`)<-[:PARENT_COMMUNITY*]-(:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount
"""

GET_COMMUNITY_INFO = """
MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e)
WHERE c.level = 0
WITH c, collect(e) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n IN nodes | {
          id: n.id,
          description: n.description,
          element_summary: n.element_summary,
          type: [el IN labels(n) WHERE el <> '__Entity__'][0],
          degree: size([(n)--() | 1])
       }] AS nodes,
       [r IN relationships | {
          start: startNode(r).id,
          type: type(r),
          end: endNode(r).id,
          description: r.description,
          combined_degree: size([(startNode(r))--() | 1]) + size([(endNode(r))--() | 1])
       }] AS rels,
       [(n)-[:HAS_CLAIM]->(cl:__Claim__) WHERE n IN nodes | {
          id: cl.id,
          subject: cl.subject,
          object: cl.object,
          type: cl.type,
          description: cl.description,
          source_span: cl.source_span,
          start_date: cl.start_date,
          end_date: cl.end_date,
          entity_id: n.id,
          entity_ids: [(e2)-[:HAS_CLAIM]->(cl) WHERE e2 IN nodes | e2.id]
       }] AS claims
"""

# Leaf community element packs under a specific parent (for level-1 substitution).
GET_LEAF_ELEMENTS_UNDER_PARENT = """
MATCH (p:`__Community__` {id: $parent_id})<-[:PARENT_COMMUNITY]-(leaf:`__Community__` {level: 0})
MATCH (leaf)<-[:IN_COMMUNITY]-(e)
WITH p, leaf, collect(e) AS nodes
WHERE size(nodes) > 0
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN leaf.id AS communityId,
       leaf.summary AS summary,
       coalesce(leaf.title, '') AS title,
       [n IN nodes | {
          id: n.id,
          description: n.description,
          element_summary: n.element_summary,
          type: [el IN labels(n) WHERE el <> '__Entity__'][0],
          degree: size([(n)--() | 1])
       }] AS nodes,
       [r IN relationships | {
          start: startNode(r).id,
          type: type(r),
          end: endNode(r).id,
          description: r.description,
          combined_degree: size([(startNode(r))--() | 1]) + size([(endNode(r))--() | 1])
       }] AS rels,
       [(n)-[:HAS_CLAIM]->(cl:__Claim__) WHERE n IN nodes | {
          id: cl.id,
          subject: cl.subject,
          object: cl.object,
          type: cl.type,
          description: cl.description,
          source_span: cl.source_span,
          start_date: cl.start_date,
          end_date: cl.end_date,
          entity_id: n.id,
          entity_ids: [(e2)-[:HAS_CLAIM]->(cl) WHERE e2 IN nodes | e2.id]
       }] AS claims
"""

SET_PAPER_COMMUNITY_LEVELS = """
MATCH (c:__Community__)
WITH max(c.level) AS max_level
MATCH (c2:__Community__)
SET c2.paper_level = max_level - c2.level,
    c2.paper_level_label = 'C' + toString(max_level - c2.level)
"""

GET_MAX_STORED_COMMUNITY_LEVEL = """
MATCH (c:__Community__)
RETURN coalesce(max(c.level), 0) AS max_level
"""

# Direct children only — parents are summarized bottom-up one Leiden level at a time.
GET_PARENT_COMMUNITY_INFO_AT_LEVEL = """
MATCH (p:`__Community__` {level: $level})<-[:PARENT_COMMUNITY]-(c:`__Community__`)
WHERE p.summary IS NULL AND c.summary IS NOT NULL
WITH p, collect({
  id: c.id,
  summary: c.summary,
  title: coalesce(c.title, ''),
  level: c.level
}) AS children
WHERE size(children) > 0
RETURN p.id AS communityId, p.level AS level, children
"""

# Legacy query kept for compatibility / tests that inspect module attributes.
GET_PARENT_COMMUNITY_INFO = """
MATCH (p:`__Community__`)<-[:PARENT_COMMUNITY*]-(c:`__Community__`)
WHERE p.summary is null and c.summary is not null
RETURN p.id as communityId, collect(c.summary) as texts
"""


STORE_COMMUNITY_SUMMARIES = """
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.summary = row.summary,
    c.title = row.title
""" 


COMMUNITY_SYSTEM_TEMPLATE = "Given input triples and optional claim covariates, generate the information summary. No pre-amble."


COMMUNITY_TEMPLATE = """
Based on the provided nodes, relationships, and claim covariates that belong to the same graph community,
generate following output in exact format
title: A concise title, no more than 4 words,
summary: A natural language summary of the information. Incorporate relevant claims/covariates when present.
{community_info}
Example output:
title: Example Title,
summary: This is an example summary that describes the key information of this community.
"""

PARENT_COMMUNITY_SYSTEM_TEMPLATE = "Given an input list of community summaries, generate a summary of the information"

PARENT_COMMUNITY_TEMPLATE = """Based on the provided list of community summaries that belong to the same graph community, 
generate following output in exact format
title: A concise title, no more than 4 words,
summary: A natural language summary of the information. Include all the necessary information as much as possible.

{community_info}

Example output:
title: Example Title,
summary: This is an example summary that describes the key information of this community.
""" 


GET_COMMUNITY_DETAILS = """
MATCH (c:`__Community__`)
WHERE  c.embedding IS NULL AND c.summary IS NOT NULL
RETURN c.id as communityId, c.summary as text
"""

WRITE_COMMUNITY_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c) WHERE c.id = row.communityId
CALL db.create.setNodeVectorProperty(c, "embedding", row.embedding)
"""  

DROP_COMMUNITIES = "MATCH (c:`__Community__`) DETACH DELETE c"
DROP_COMMUNITY_PROPERTY = "MATCH (e:`__Entity__`) REMOVE e.communities"


ENTITY_VECTOR_INDEX_NAME = "entity_vector"
ENTITY_VECTOR_EMBEDDING_DIMENSION = 384

DROP_ENTITY_VECTOR_INDEX_QUERY = f"DROP INDEX {ENTITY_VECTOR_INDEX_NAME} IF EXISTS;"
CREATE_ENTITY_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (e:__Entity__) ON e.embedding
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
""" 

COMMUNITY_VECTOR_INDEX_NAME = "community_vector"
COMMUNITY_VECTOR_EMBEDDING_DIMENSION = 384

DROP_COMMUNITY_VECTOR_INDEX_QUERY = f"DROP INDEX {COMMUNITY_VECTOR_INDEX_NAME} IF EXISTS;"
CREATE_COMMUNITY_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (c:__Community__) ON c.embedding
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
""" 

COMMUNITY_FULLTEXT_INDEX_NAME = "community_keyword"
COMMUNITY_FULLTEXT_INDEX_DROP_QUERY = f"DROP INDEX  {COMMUNITY_FULLTEXT_INDEX_NAME} IF EXISTS;"
COMMUNITY_INDEX_FULL_TEXT_QUERY = f"CREATE FULLTEXT INDEX {COMMUNITY_FULLTEXT_INDEX_NAME} FOR (n:`__Community__`) ON EACH [n.summary]" 



def get_gds_driver(uri, username, password, database):
    try:
        if all(v is None for v in [username, password]):
            username= get_value_from_env('NEO4J_USERNAME')
            database= get_value_from_env('NEO4J_DATABASE')
            password= get_value_from_env('NEO4J_PASSWORD')
            
        gds = GraphDataScience(
            endpoint=uri,
            auth=(username, password),
            database=database
        )
        logging.info("Successfully created GDS driver.")
        return gds
    except Exception as e:
        logging.error(f"Failed to create GDS driver: {e}")
        raise

def create_community_graph_projection(gds, project_name=COMMUNITY_PROJECTION_NAME, node_projection=NODE_PROJECTION):
    try:
        existing_projects = gds.graph.list()
        project_exists = existing_projects["graphName"].str.contains(project_name, regex=False).any()
        
        if project_exists:
            logging.info(f"Projection '{project_name}' already exists. Dropping it.")
            gds.graph.drop(project_name)
        
        logging.info(f"Creating new graph project '{project_name}'.")
        projection_query = CREATE_COMMUNITY_GRAPH_PROJECTION.format(node_projection=node_projection,project_name=project_name)
        graph_projection_result = gds.run_cypher(projection_query)
        projection_result = graph_projection_result.to_dict(orient="records")[0]
        logging.info(f"Graph projection '{projection_result['graph_name']}' created successfully with {projection_result['nodes']} nodes and {projection_result['rels']} relationships.")
        graph_project = gds.graph.get(projection_result['graph_name'])
        return graph_project
    except Exception as e:
        logging.error(f"Failed to create community graph project: {e}")
        raise

def write_communities(gds, graph_project, project_name=COMMUNITY_PROJECTION_NAME):
    try:
        logging.info(f"Writing communities to the graph project '{project_name}'.")
        gds.leiden.write(
            graph_project,
            writeProperty=project_name,
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
            maxLevels=MAX_COMMUNITY_LEVELS,
            minCommunitySize=MIN_COMMUNITY_SIZE,
        )
        logging.info("Communities written successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to write communities: {e}")
        return False


def get_community_chain(llm, is_parent=False,community_template=COMMUNITY_TEMPLATE,system_template=COMMUNITY_SYSTEM_TEMPLATE):
    try:
        if is_parent:
            community_template=PARENT_COMMUNITY_TEMPLATE
            system_template= PARENT_COMMUNITY_SYSTEM_TEMPLATE
        community_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_template,
                ),
                ("human", community_template),
            ]
        )

        community_chain = community_prompt | llm | StrOutputParser()
        return community_chain
    except Exception as e:
        logging.error(f"Failed to create community chain: {e}")
        raise

def prepare_string(community_data, max_chars=12000):
    """
    Build leaf-community prompt text with GraphRAG-style degree prioritization.
    Higher combined-degree relationships are included first until the token/char budget is reached.
    """
    try:
        from src.graphrag.helpers import prepare_community_string

        return prepare_community_string(community_data, max_chars=max_chars)
    except Exception as e:
        logging.error(f"Failed to prepare string from community data: {e}")
        raise

def process_community_info(community, chain, is_parent=False):
    try:
        if is_parent:
            # Prefer pre-packed GraphRAG substitution context when available.
            combined_text = community.get("packed_context") or ""
            if not combined_text:
                texts = community.get("texts") or []
                if texts and isinstance(texts[0], dict):
                    combined_text = " ".join(
                        f"Summary {i+1}: {item.get('summary') or ''}" for i, item in enumerate(texts)
                    )
                else:
                    combined_text = " ".join(
                        f"Summary {i+1}: {summary}" for i, summary in enumerate(texts)
                    )
        else:
            combined_text = prepare_string(community)
        summary_response = chain.invoke({'community_info': combined_text})
        lines = summary_response.splitlines()
        title = "Untitled Community"
        summary = ""
        for line in lines:
            if line.lower().startswith("title"):
                title = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("summary"):
                summary = line.split(":", 1)[-1].strip()     
        logging.info(f"Community Title : {title}")
        return {"community": community['communityId'], "title":title, "summary": summary}
    except Exception as e:
        logging.error(f"Failed to process community {community.get('communityId', 'unknown')}: {e}")
        return None


def _parent_token_budget() -> int:
    return get_value_from_env("GRAPHRAG_PARENT_SUMMARY_TOKEN_BUDGET", 3000, "int")


def _build_parent_subcommunities(gds, parent_row: dict) -> list:
    """
    Attach element_text packs for leaf children when available so higher-level
    summarization can substitute sub-community summaries under the token budget.
    """
    from src.graphrag.helpers import prepare_community_string

    children = list(parent_row.get("children") or [])
    parent_id = parent_row.get("communityId")
    parent_level = parent_row.get("level")
    # Level 1 parents have leaf children — fetch element packs for substitution.
    element_by_id = {}
    if parent_level == 1 and parent_id:
        try:
            leaf_rows = gds.run_cypher(
                GET_LEAF_ELEMENTS_UNDER_PARENT, params={"parent_id": parent_id}
            )
            for leaf in leaf_rows.to_dict(orient="records"):
                element_by_id[leaf["communityId"]] = prepare_community_string(leaf)
        except Exception as exc:
            logging.warning(
                "Could not fetch leaf elements for parent %s: %s", parent_id, exc
            )

    subcommunities = []
    for child in children:
        child_id = child.get("id")
        subcommunities.append(
            {
                "id": child_id,
                "summary": child.get("summary") or "",
                "title": child.get("title") or "",
                "element_text": element_by_id.get(child_id, ""),
            }
        )
    return subcommunities


def create_community_summaries(gds, model, email, uri):
    callback_handler = None
    token_usage = 0
    try:
        #pre check if user allowed to create community summaries
        if get_value_from_env("TRACK_USER_USAGE", "false", "bool"):
            try:
                track_token_usage(email, uri, 0, model,operation_type="precheck")
            except LLMGraphBuilderException as e:
                logging.error(str(e))
                raise RuntimeError(str(e))
        community_info_list = gds.run_cypher(GET_COMMUNITY_INFO)
        llm, model_name,callback_handler = get_llm(model)
        community_chain = get_community_chain(llm)
        
        summaries = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_community_info, community, community_chain) for community in community_info_list.to_dict(orient="records")]
   
            for future in as_completed(futures):
                result = future.result()
                if result:
                    summaries.append(result)
                else:
                    logging.error("community summaries could not be processed.")

        if summaries:
            gds.run_cypher(STORE_COMMUNITY_SUMMARIES, params={"data": summaries})

        # Bottom-up parent summarization with GraphRAG token-budget substitution.
        from src.graphrag.helpers import build_higher_level_community_context

        max_level_rows = gds.run_cypher(GET_MAX_STORED_COMMUNITY_LEVEL)
        max_level = 0
        if not max_level_rows.empty:
            max_level = int(max_level_rows.iloc[0].get("max_level") or 0)

        parent_community_chain = get_community_chain(llm, is_parent=True)
        budget = _parent_token_budget()

        for level in range(1, max_level + 1):
            parent_rows = gds.run_cypher(
                GET_PARENT_COMMUNITY_INFO_AT_LEVEL, params={"level": level}
            )
            parent_records = parent_rows.to_dict(orient="records")
            if not parent_records:
                continue

            prepared = []
            for parent in parent_records:
                subcommunities = _build_parent_subcommunities(gds, parent)
                packed, meta = build_higher_level_community_context(
                    subcommunities, max_tokens=budget
                )
                if not packed:
                    # Fallback: concatenate child summaries
                    packed = " ".join(
                        f"Summary {i+1}: {c.get('summary') or ''}"
                        for i, c in enumerate(parent.get("children") or [])
                    )
                parent = dict(parent)
                parent["packed_context"] = packed
                parent["pack_meta"] = meta
                prepared.append(parent)
                logging.info(
                    "Parent %s level=%s pack substitutions=%s tokens≈%s",
                    parent.get("communityId"),
                    level,
                    meta.get("substitutions"),
                    meta.get("tokens"),
                )

            parent_summaries = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_community_info, community, parent_community_chain, True
                    )
                    for community in prepared
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        parent_summaries.append(result)
                    else:
                        logging.error("parent community summaries could not be processed.")

            if parent_summaries:
                gds.run_cypher(STORE_COMMUNITY_SUMMARIES, params={"data": parent_summaries})

    except Exception as e:
        logging.error(f"Failed to create community summaries: {e}")
        raise

    finally:
       try:
           if get_value_from_env("TRACK_USER_USAGE", "false", "bool"):
               if callback_handler:
                   usage = callback_handler.report()
                   token_usage = usage.get("total_tokens", 0)
                   if email and token_usage > 0:
                       email = email.strip().replace('"', '')
                       latest_token = track_token_usage(email, uri, token_usage, model,operation_type="community_summary")
                       logging.info(f"In community : Total token usage {latest_token} for user {email} ")
       except Exception as err:
           logging.warning(f"Failed to track token usage: {err}")

def create_community_embeddings(gds, embedding_provider, embedding_model):
    try:
        embeddings, dimension = load_embedding_model(embedding_provider, embedding_model)
        logging.info(f"Embedding model '{embedding_model}' loaded successfully.")
        
        logging.info("Fetching community details.")
        rows = gds.run_cypher(GET_COMMUNITY_DETAILS)
        rows = rows[['communityId', 'text']].to_dict(orient='records')
        logging.info(f"Fetched {len(rows)} communities.")
        
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i+batch_size]            
            for row in batch_rows:
                try:
                    row['embedding'] = embeddings.embed_query(row['text'])
                except Exception as e:
                    logging.error(f"Failed to embed text for community ID {row['communityId']}: {e}")
                    row['embedding'] = None
            
            try:
                logging.info("Writing embeddings to the database.")
                gds.run_cypher(WRITE_COMMUNITY_EMBEDDINGS, params={'rows': batch_rows})
                logging.info("Embeddings written successfully.")
            except Exception as e:
                logging.error(f"Failed to write embeddings to the database: {e}")
                continue
        return dimension
    except Exception as e:
        logging.error(f"An error occurred during the community embedding process: {e}")


def create_vector_index(gds, index_type,embedding_dimension=None):
    drop_query = ""
    query = ""
    
    if index_type == ENTITY_VECTOR_INDEX_NAME:
        drop_query = DROP_ENTITY_VECTOR_INDEX_QUERY
        query = CREATE_ENTITY_VECTOR_INDEX_QUERY.format(
            index_name=ENTITY_VECTOR_INDEX_NAME,
            embedding_dimension=embedding_dimension if embedding_dimension else ENTITY_VECTOR_EMBEDDING_DIMENSION
        )
    elif index_type == COMMUNITY_VECTOR_INDEX_NAME:
        drop_query = DROP_COMMUNITY_VECTOR_INDEX_QUERY
        query = CREATE_COMMUNITY_VECTOR_INDEX_QUERY.format(
            index_name=COMMUNITY_VECTOR_INDEX_NAME,
            embedding_dimension=embedding_dimension if embedding_dimension else COMMUNITY_VECTOR_EMBEDDING_DIMENSION
        )
    else:
        logging.error(f"Invalid index type provided: {index_type}")
        return

    try:
        logging.info("Starting the process to create vector index.")

        logging.info(f"Executing drop query: {drop_query}")
        gds.run_cypher(drop_query)

        logging.info(f"Executing create query: {query}")
        gds.run_cypher(query)

        logging.info(f"Vector index '{index_type}' created successfully.")
    
    except Exception as e:
        logging.error("An error occurred while creating the vector index.", exc_info=True)
        logging.error(f"Error details: {str(e)}")


def create_fulltext_index(gds, index_type):
    drop_query = ""
    query = ""
    
    if index_type == COMMUNITY_FULLTEXT_INDEX_NAME:
        drop_query = COMMUNITY_FULLTEXT_INDEX_DROP_QUERY
        query = COMMUNITY_INDEX_FULL_TEXT_QUERY
    else:
        logging.error(f"Invalid index type provided: {index_type}")
        return

    try:
        logging.info("Starting the process to create full-text index.")

        logging.info(f"Executing drop query: {drop_query}")
        gds.run_cypher(drop_query)

        logging.info(f"Executing create query: {query}")
        gds.run_cypher(query)

        logging.info(f"Full-text index '{index_type}' created successfully.")

    except Exception as e:
        logging.error("An error occurred while creating the full-text index.", exc_info=True)
        logging.error(f"Error details: {str(e)}")

def create_community_properties(gds, model, email, uri, embedding_provider, embedding_model):
    commands = [
        (CREATE_COMMUNITY_CONSTRAINT, "created community constraint to the graph."),
        (CREATE_COMMUNITY_LEVELS, "Successfully created community levels."),
        (CREATE_COMMUNITY_RANKS, "Successfully created community ranks."),
        (CREATE_PARENT_COMMUNITY_RANKS, "Successfully created parent community ranks."),
        (CREATE_COMMUNITY_WEIGHTS, "Successfully created community weights."),
        (CREATE_PARENT_COMMUNITY_WEIGHTS, "Successfully created parent community weights."),
        (SET_PAPER_COMMUNITY_LEVELS, "Successfully mapped paper community levels C0-C3."),
    ]
    try:
        for command, message in commands:
            gds.run_cypher(command)
            logging.info(message)

        create_community_summaries(gds, model, email, uri)
        logging.info("Successfully created community summaries.")

        embedding_dimension = create_community_embeddings(gds, embedding_provider, embedding_model)
        logging.info("Successfully created community embeddings.")

        create_vector_index(gds=gds,index_type=ENTITY_VECTOR_INDEX_NAME,embedding_dimension=embedding_dimension)
        logging.info("Successfully created Entity Vector Index.")

        create_vector_index(gds=gds,index_type=COMMUNITY_VECTOR_INDEX_NAME,embedding_dimension=embedding_dimension)
        logging.info("Successfully created community Vector Index.")

        create_fulltext_index(gds=gds,index_type=COMMUNITY_FULLTEXT_INDEX_NAME)
        logging.info("Successfully created community fulltext Index.")

    except Exception as e:
        logging.error(f"Error during community properties creation: {e}")
        raise


def clear_communities(gds):
    try:
        logging.info("Starting to clear communities.")

        logging.info("Dropping communities...")
        gds.run_cypher(DROP_COMMUNITIES)
        logging.info(f"Communities dropped successfully")

        logging.info("Dropping community property from entities...")
        gds.run_cypher(DROP_COMMUNITY_PROPERTY)
        logging.info(f"Community property dropped successfully")

    except Exception as e:
        logging.error(f"An error occurred while clearing communities: {e}")
        raise


def create_communities(uri, username, password, database,email=None,model=COMMUNITY_CREATION_DEFAULT_MODEL, embedding_provider=None, embedding_model=None):
    try:
        gds = get_gds_driver(uri, username, password, database)
        clear_communities(gds)

        graph_project = create_community_graph_projection(gds)
        write_communities_success = write_communities(gds, graph_project)
        if write_communities_success:
            logging.info("Starting Community properties creation process.")
            create_community_properties(gds,model,email,uri, embedding_provider, embedding_model)
            logging.info("Communities creation process completed successfully.")
        else:
            logging.warning("Failed to write communities. Constraint was not applied.")
    except Exception as e:
        logging.error(f"Failed to create communities: {e}")
