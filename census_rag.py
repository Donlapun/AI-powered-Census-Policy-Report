"""US Census Data Policy Report.

This module create the interactive AI Dashboard for US Census Data where
user can input their questions about the US Census Data and received
the data, visualization, and policy report about findings.
"""

import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO, BytesIO
from fpdf import FPDF
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import wikipedia

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import create_agent
import os

from utils import normalize_name, fetch_census_data, fetch_shapefile, format_tables

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")


# Helper funtions that requires prompt engieering and agents


# Tool 1: Using Cene Geociding API
@tool
def census_geocode(query: str):
    """
    Given 'Place, State', returns list of counties the place belongs to using Census API
    """
    try:
        place, state = query.split(",")
        place = place.strip()
        state = state.strip()
    except:
        return {"error": "Query must be 'Place, State'"}

    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": f"{place}, {state}",
        "benchmark": "Public_AR_Census2020",
        "format": "json",
    }
    try:
        r = requests.get(url, params=params).json()
        matches = r.get("result", {}).get("addressMatches", [])
        counties = set()  # prevent duplicates
        for m in matches:
            comp = m["addressComponents"]
            counties.add(comp["county_name"])
        if counties:
            return {"counties": list(counties)}
        else:
            return {"error": "No counties found from Census API"}
    except Exception as e:
        return {"error": str(e)}


# Tool 2: Wikipedia
@tool
def wikipedia_fips(query: str):
    """
    Given 'Place, State', returnslist of counties the place from Wikipedia as a fallback.
    """
    try:
        place, state = query.split(",")
        place = place.strip()
        state = state.strip()

        # Get page content
        page = wikipedia.page(f"{place}, {state}")
        content = page.content[:1000]

        # Prompt for LLM
        prompt = f"""
        You are a geographic assistant. Extract all county names for the place below, but only return counties that belong to the {state}

        Place: {place}, {state}

        Wikipedia content:
        {content}

        Rules:
        - Return JSON only with a key "counties" whose value is a list of county names.
        - Only include counties that belong to the state {state}.       
        - Example: {{"counties": ["Cook"]}}
        - If no counties are found for this state, return {{"counties": []}}.
        """

        response = llm_parse.invoke(prompt)
        content_json = json.loads(response.content)

        return content_json

    except Exception as e:
        return {"error": str(e)}


# ChatOpenAI LLM for generating AI responses
llm_parse = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY
)

# ReAct agent for retreiving counties of geographic location
agent = create_agent(
    tools=[census_geocode, wikipedia_fips],
    model=llm_parse,
    system_prompt="""
        You are a geographic assistant that finds FIPS codes for U.S. places. 
        Always use the exact Place and State provided in the input. 
        Do not replace it with any other city or state. 
        Do not hallucinate or guess.
        You have access to the following tools:

        - Census Counties Lookup: returns a list of counties for the place in the correct state. Do NOT return counties from a different state.
        - Wikipedia Counties Lookup: returns a list of counties for the place in the correct state. Do NOT return counties from a different state.

        Rules:
        1. Always try Census first, then Wikipedia.
        2. Respond ONLY in JSON format, with no extra text or commentary.
        3. JSON keys:
        {
            "state": "<2-digit state FIPS>",
            "county": ["<3-digit county FIPS>", ...]  # list of all counties in the place
        }
        4. If no valid FIPS can be found, return {"error": "message"}.
        5. Do NOT include any explanatory text outside the JSON.
        6. Input queries will be in the format: "Place, State" .
    """,
)


def get_fips(geography_type, geography_name, year, state_name=None):
    """
    Retrieve the FIPS code(s) for a given geographic location using API and
    ReAct Agent.

    Args:
        geography_type (str): Type of geography ('state', 'county' or 'place').
        geography_name (str): Name of the location from query.
        year (int): Year requested.
        state_name (str, optional): Name of the state. Defaults to None.

    Returns:
        dict: A dictionary containing FIPS codes:
            - "state": str, FIPS code of the state
            - "county": str or list, FIPS code of the county

    Raises:
        ValueError: If the geography_type is invalid or the geography_name is not found.
    """

    if geography_type == "state":
        url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME&for=state:*"
        try:
            result = requests.get(url).json()
        except Exception as e:
            st.error(f"Failed to fetch state FIPS: {e}")
            return None

        for name, fips in result[1:]:
            if name.lower() == geography_name.lower():
                return {"state": fips.zfill(2)}

        st.error(f"Could not find state: {geography_name}")
        return None

    elif geography_type == "county":
        if state_name is None:
            st.error("State name is required for county FIPS lookup.")
            return None

        # First get state FIPS
        state_fips_dict = get_fips("state", state_name, year)
        if not state_fips_dict:
            return None
        state_fips = state_fips_dict["state"]

        # Get counties in the state
        url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME&for=county:*&in=state:{state_fips}"
        try:
            result = requests.get(url).json()
        except Exception as e:
            st.error(f"Failed to fetch county FIPS: {e}")
            return None

        for name, _, county_fips in result[1:]:
            if normalize_name(name.split(",")[0].strip()) == normalize_name(
                geography_name
            ):
                return {"state": state_fips, "county": county_fips.zfill(3)}

        st.error(f"Could not find county: {geography_name} in state: {state_name}")
        return None

    elif geography_type == "place":
        if state_name is None:
            st.error("State name is required for place FIPS lookup.")
            return None

        # Get counties in the state
        state_fips_dict = get_fips("state", state_name, year)
        if not state_fips_dict:
            return None

        state_fips = state_fips_dict["state"]

        # Using ReAct agent to try differnt tools
        place_query = f"{geography_name}, {state_name}"

        try:
            result = agent.invoke({"input": place_query}, return_only_outputs=True)

            # Retrieve Result
            for r in result["messages"]:
                content = getattr(r, "content", None)
                if content:
                    try:
                        data = json.loads(content)
                        if "counties" in data:
                            counties = data["counties"]
                            break  # stop once we find the counties
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in message: {content}")

            # Parse JSON
            fips_data = json.loads(content)

            fips_data["state"] = state_fips
            county_fips = []
            for c in fips_data["counties"]:
                fips = get_fips("county", c, year, state_name)
                if fips:
                    county_fips.append(fips["county"].zfill(3))

            return {"state": state_fips, "county": county_fips}

        except json.JSONDecodeError:
            st.error(f"Agent returned invalid JSON: {content}")
            return None
        except Exception as e:
            st.error(f"Agent failed: {e}")
            return None

    else:
        st.error(f"Unknown geography_type: {geography_type}")
        return None


def clean_variable(df, year):
    variables_url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
    resp = requests.get(variables_url)
    variables_json = resp.json()["variables"]

    var_info = {}
    for var_id in df.columns:
        if var_id in variables_json:
            var_info[var_id] = {
                "concept": variables_json[var_id].get("concept", ""),
                "label": variables_json[var_id].get("label", ""),
            }

    column_prompt = """
        You are a data labeling assistant. 
        Create a **concise, human-readable label** for a Census variable.

        Here is the dictionary of Census variables: {var_info}

        Output rules:
        - Return a short, human-readable label for each variable in the format: CLEANED_LABEL - VAR_ID
        - Use parentheses for subcategories, e.g., "White (Not Hispanic/Latino)".
        - For totals (_001E), use descriptive names like "Total Population".
        - Return one line per variable, **do NOT skip any variables**.
        - Do NOT return JSON, only plain text.
        """
    response = llm_parse.invoke(column_prompt.format(var_info=var_info))
    response_text = response.content.strip()
    cleaned_labels = {}
    for line in response_text.splitlines():
        if " - " in line:
            label, var_id = line.split(" - ")
            cleaned_labels[var_id.strip()] = line.strip()

    df_clean = df.rename(columns=lambda col: cleaned_labels.get(col, col))

    return df_clean


# Initialization
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "user_query" not in st.session_state:
    st.session_state.user_query = None

# Streamlit Interface
st.title("AI-powered US Census Data Auto Report Policy")

st.write("""
The U.S. Census Bureau collects detailed data about people and communities.
Common topics include population, age, race & ethnicity, income, housing, education, and employment.
Data is available at many geographic levels—nation, state, county, city and census tract.
""")

st.subheader("Why it’s useful")
st.write("""
Helps researchers, policymakers, nonprofits, and businesses understand demographic patterns and community needs.
Informs planning and allocation of resources (e.g., federal funding, services).
Enables analysis of social and economic disparities and tracking trends over time.
""")

st.subheader("Your Census Policy Report")
st.write("""
To generate the report, just provide a few details:
- Location: State, county, or place you want data for.
- Time: Year
- Topic: The type of data you’re interested in (e.g., population, race, employment, housing).
         
Our AI will handle the rest — no need to manually search or compile data yourself!”
""")

user_query = st.text_input(
    "Example question: Race disparity in New York in 2019",
    "Race disparity in New York in 2019",
)

if st.button("Submit"):
    st.session_state.submitted = True
    st.session_state.user_query = user_query

if st.session_state.submitted:
    parse_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
            Extract the following fields from the user's query:

            - topic
            - geography_type: one of ["state", "county", "place"]
            - geography_name
            - state_name (required if geography_type is "county" or "place")
            - year (4-digit). If not provided, default to 2021.
            - Return ONLY valid JSON. No explanation.

            Rules:
            - If the query mentions a U.S. state by name, include it in state_name.
            - Do NOT treat states as cities.
            - If the geography is a county or place, state_name must be included for FIPS lookup.
            - If a place is missing but a state is present, return the state.
            - Never guess or substitute defaults.
            - Never output Chicago or Cook County unless explicitly mentioned.


            Question: {question}
        """,
    )
    parsed_json = llm_parse.invoke(parse_prompt.format(question=user_query))
    raw = parsed_json.content.strip()

    try:
        parsed = json.loads(raw)
    except:
        st.error(f"Failed to parse question. Raw output: {parsed_json}")
        st.stop()

    topic = parsed.get("topic")
    geography_type = parsed.get("geography_type")
    geography_name = parsed.get("geography_name")
    state_name = parsed.get("state_name")
    year = str(parsed.get("year"))

    st.info(
        f"Parsed Topic: {topic}, Geography: {geography_type} = {geography_name}, State: {state_name}, Year: {year}"
    )

    fips = get_fips(geography_type, geography_name, year, state_name)

    table_prompt = PromptTemplate(
        input_variables=["topic"],
        template="""
        You are an expert in U.S. Census ACS 5-year data.
        For the topic {topic}, ONLY output valid JSON in the following format:
        {{"tables": ["tableID1", "tableID2", ...]}}
        Do NOT include any text, explanation, or extra characters before or after the JSON.
        Do not add newlines, comments, or quotes outside the JSON object.
    
        """,
    )
    tables_json = llm_parse.invoke(table_prompt.format(topic=topic))

    try:
        tables = json.loads(tables_json.content).get("tables", ["B02001"])
    except:
        tables = ["B02001"]

    variables_str = format_tables(tables)

    df = fetch_census_data(year, variables_str, geography_type, geography_name, fips)

    # Convert to numerical columns
    numeric_cols = [
        col
        for col in df.columns
        if col not in ["NAME", "state", "county", "tract", "GEOID"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = clean_variable(df, year)

    st.subheader("Census Data")
    st.dataframe(df)

    st.subheader("Summary Statistics of Data")
    summary_stat = df.describe()
    st.dataframe(summary_stat)

    st.subheader("Census Visualizations")
    if geography_type == "state":
        fips["county"] = None
    state_shapefile = fetch_shapefile(
        geography_type, fips["state"], fips["county"], year
    )

    geo_df = state_shapefile[["GEOID", "geometry"]].merge(df, on="GEOID", how="left")

    st.session_state["gdf"] = geo_df

    # Visualizations for Census Data
    if "gdf" in st.session_state:
        gdf = st.session_state["gdf"]

        choro_columns = [
            col
            for col in gdf.columns
            if col not in ["NAME", "state", "county", "tract", "GEOID", "geometry"]
        ]
        user_metric = st.selectbox("Choose a variable:", choro_columns)

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3
        )

        # Left column: map spans both rows
        ax_map = fig.add_subplot(gs[:, 0])
        gdf.plot(
            column=user_metric,
            ax=ax_map,
            legend=True,
            cmap="viridis",
            edgecolor="grey",
            linewidth=0.3,
            legend_kwds={
                "label": "Estimate Total",
                "orientation": "horizontal",
                "shrink": 0.8,
                "pad": 0.05,
            },
        )
        ax_map.set_title(f"Choropleth Map of {user_metric}", fontsize=16)
        ax_map.axis("off")

        # Upper Right column, top: histogram
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_hist.hist(
            gdf[user_metric],
            bins=25,
            edgecolor="black",
            color="green",
        )
        ax_hist.set_title(f"Histogram of {user_metric}", fontsize=14)
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_xlabel(user_metric)
        ax_hist.tick_params(axis="x", rotation=45)

        # Lower Right column, bottom: heatmap
        ax_corr = fig.add_subplot(gs[1, 1])

        corr_matrix = gdf[choro_columns].corr()
        short_label = [col.split(" - ")[0].strip() for col in choro_columns]
        short_label = [
            label[:30] if len(label) > 30 else label for label in short_label
        ]
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            ax=ax_corr,
            xticklabels=short_label,
            yticklabels=short_label,
        )

        ax_corr.set_title("Correlation Matrix", fontsize=14)

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

        st.pyplot(fig)

        # Download option for graph
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.download_button(
            label="Download Figure as PNG",
            data=buf,
            file_name="visualization.png",
            mime="image/png",
        )

    else:
        st.warning("No census data loaded yet.")

    st.header("Policy Report")

    # Multiple Agents for Policy Report
    @tool
    def data_agent(df_head, summary_stats, corr_matrix):
        "Data Analysis Agent"
        report_prompt = PromptTemplate(
            input_variables=["df_head", "summary_stats", "corr_matrix"],
            template="""
                You are a data analyst for  policy insights.

                Tasks: 
                - Identify the top and bottom areas (GEOIDs) for each variable and interpret their significance.
                - Highlight disparities, gaps, or notable trends in the data.
                - Generate Human-readable insights for policy makers
                - Suggest metrics or indicators for further analysis 

                Response:
                - summary of the data
                - insights: list of text insights
                - flags: any disparities or notable trends

                Data Provided:
                Sample Data (first 5 rows): {df_head}
                Summary Statistics: {summary_stats}
                Corrlation MAtrix: {corr_matrix}
            """,
        )
        data_report = llm_parse.invoke(
            report_prompt.format(
                df_head=df,
                summary_stats=summary_stat,
                corr_matrix=corr_matrix,
            )
        )
        return data_report.content

    @tool
    def evidence_agent(trend_insights):
        "Evidence Supporting Agent"
        evidence_prompt = """
            You are an evidence retrieval agent.
            You are given the following trends/insights: {trend_insights}

            Tasks:
            1. For each insight, search ONLY reliable public sources 
            (Urban Institute, Census.gov, ACS tables, BLS, CDC, academic papers).
            

            2. First VALIDATE that data for the required years actually exists.
            - If a year is missing or the exact metric is unavailable, find new source. 
            - Do NOT invent URLs.
            - The year difference must not be more than 1 year.

            3. Retrieve factual evidence that supports or contradicts the insight.
            Summaries should be 1–2 sentences max.

            4. Return results in this exact structure:
            [
                {{
                    "insight": "...",
                    "evidence_summary": "...",
                    "data_year_used": "...",
                    "sources": [
                        {{"title": "...", "url": "..."}}
                    ]
                }}
            ]

            """
        evidence_report = llm_parse.invoke(
            evidence_prompt.format(
                trend_insights=trend_insights,
            )
        )
        return evidence_report.content

    @tool
    def policy_report_agent(trend_insights, evidence_report):
        "Policy Report Agent"
        policy_prompt = """
        You are a report writer. Your audience are Policymakers and stakeholders with limited technical expertise in data analysis who need evidence-based guidance.

        Trends: {trend_insights}
        Evidence: {evidence_report}

        Each evidence item has this structure:
        {{
            "insight": "...",
            "evidence_summary": "...",
            "data_year_used": "...",
            "sources": [
                {{ "title": "...", "url": "..." }}
            ]
        }}

        Tasks:
        - Use every item in the evidence and cite every source (titel, URL).
        - Alway double check the reference that it is correct and accessible
        - Format a human-readable policy report.
        - Include headings, bullets, and step-by-step explanations.
        - Ensure formatting is consistent.
        - Create recommendations based on trends and evidence.
        - compiles everything into a human-readable policy report.
        """
        policy_report = llm_parse.invoke(
            policy_prompt.format(
                trend_insights=trend_insights, evidence_report=evidence_report
            )
        )
        return policy_report.content

    policy_agent = create_agent(
        tools=[data_agent, evidence_agent, policy_report_agent],
        model=llm_parse,
        system_prompt="""
        You are a policy assistant that analyzes U.S. Census data and generates actionable policy reports.

        Rules:
        1. Always follow the sequence in chronological order: 
            - First is trend/daata insights
            - Second is Evidence supporting each trend
            - Finally, actional policy recomendations
        2. Use only the tools provided in `policy_tol`.
        3. Do not hallucinate or skip steps.
        4. Each tool should be used only for its designated purpose.
        5. When creating the final report, ensure:
            - Each trend is explicitly linked to support evidences 
            - Explain why evidence supports the trend
            - Each section is clearly connected; do not jump abruptly between topics.
            - Elaborate each section
            - Insights are explicitly linked to supporting evidence and policy recommendations.
            - Use headings and bullets where appropriate to improve readability.
            - Keep the language concise, actionable, and easy to read for policymakers.

        Tools:
        - Trend Agent: analyzes dataset and outputs trends/disparities.
        - Evidence Agent: finds supporting evidence for trends.
        - Policy Agent: generates actionable policy recommendations.
        """,
    )
    final_report = policy_agent.invoke(
        {"df_head": df, "summary_stats": summary_stat, "corr_matrix": str(corr_matrix)},
        return_only_outputs=True,
    )

    tool_messages = [m for m in final_report["messages"] if m.type == "tool"]
    print("-2", tool_messages[-2].content)

    # Assume policy_report_agent is last
    policy_report_text = tool_messages[-1].content

    st.write(policy_report_text)

    # Download option for Policy Report
    st.download_button(
        label="Download Policy Report",
        data=policy_report_text,
        file_name="policy_report.txt",
        mime="text/plain",
    )
