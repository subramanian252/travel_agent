from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from typing import Annotated, List, TypedDict
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage

import operator
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import sys
import asyncio

from test_mcp_client import get_tavily_tools
from test_mcp_client import get_airport_tools
from test_mcp_client import get_weather_tools

import asyncio
import selectors


sys.path.append(r"G:\ml projects\travel_multiagent")

from flight_tool import search_flights
from tool import tavily_search

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


load_dotenv()


def get_database_url():

    database_url = os.getenv("EXTERNAL_END_POINT")

    if not database_url:
        raise ValueError(
            "INTERNAL_DATABASE_URL environment variable is not set"
        )

    if "sslmode=" not in database_url:
        sep = "&" if "?" in database_url else "?"
        database_url += f"{sep}sslmode=require"

    return database_url


class IternaryState(TypedDict):

    messages: Annotated[List[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    weather_results: str

    iternary: str

    llm_calls: int

    flight_started: bool

    hotel_started: bool

    weather_started: bool


async def main():

    tavily_tools = await get_tavily_tools()
    airport_tools = await get_airport_tools()
    weather_tools = await get_weather_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        stream_usage=True
    )

    useful_airport_tools = [
            tool for tool in airport_tools
            if tool.name in {
                "list_airports",
                "list_routes",
                "list_airlines"
            }
        ]

    flight_tools = useful_airport_tools + [search_flights]

    flight_llm = llm.bind_tools(flight_tools)

    flight_tool_node = ToolNode(flight_tools)

    tavily_llm = llm.bind_tools(tavily_tools)

    tavily_tool_node = ToolNode(tavily_tools)

    weather_llm = llm.bind_tools(weather_tools)

    weather_tool_node = ToolNode(weather_tools)



    async def flight_node(state: IternaryState):

        if not state.get("flight_started", False):

            response = await flight_llm.ainvoke([

                SystemMessage(
                    content="""
                            You are the flight-search agent.

                            Your job is to find flight options between the user's origin
                            and destination.

                            Tool rules:
                            - ALWAYS use the `search_flights` tool for actual flight searches.
                            - DO NOT use `list_airlines` to infer or recommend flight options.
                            - DO NOT use `list_airports` unless you genuinely need to resolve
                            an airport or IATA code.
                            - DO NOT use `list_routes` for flight search.
                            - If a tool returns an error or restricted-access response,
                            do not replace it with unrelated airline or airport data.
                            Instead, use `search_flights`.

                            Infer the appropriate IATA airport codes yourself when possible.

                            Do not ask for travel dates because the available
                            `search_flights` tool does not require dates.

                            Only report flight options that come from `search_flights`.
                            """
                ),

                HumanMessage(
                    content=state["user_query"]
                )

            ])

            result = {

                "llm_calls": state.get("llm_calls", 0) + 1,

                "messages": [response],

                "flight_started": True

            }

            return result


        response = await flight_llm.ainvoke(

            state["messages"]

        )

        result = {

            "messages": [response],

            "llm_calls": state.get("llm_calls", 0) + 1

        }

        if not response.tool_calls:

            result["flight_results"] = response.content

        return result


    async def hotel_node(state: IternaryState):

        query = f"best hotels in {state['user_query']}"

        if not state.get("hotel_started", False):

            response = await tavily_llm.ainvoke([

                SystemMessage(
                    content="You are a hotel search agent. Use the tavily_search tool to find hotels."
                ),

                HumanMessage(
                    content=query
                )

            ])

            return {

                "messages": [response],

                "hotel_started": True,

                "llm_calls": state.get("llm_calls", 0) + 1

            }


        response = await tavily_llm.ainvoke(

            state["messages"]

        )

        result = {

            "messages": [response],

            "llm_calls": state.get("llm_calls", 0) + 1

        }

        if not response.tool_calls:

            result["hotel_results"] = response.content

        return result

    
    async def weather_node(state: IternaryState):
        query = f"weather in {state['user_query']}"

        if not state.get("weather_started", False):
            response = await weather_llm.ainvoke([
                SystemMessage(
                        content="""
                                You are a weather research agent for a travel-planning system.

                                Your job is to gather weather information for the user's destination
                                so the itinerary can be adapted to expected conditions.

                                Tool usage:
                                - Use `get_forecast` for the destination city when planning an upcoming trip.
                                - Use `get_weather` when current weather conditions are useful.
                                - You may call both tools when both current conditions and forecast data
                                would improve the travel plan.
                                - Focus primarily on the destination city; do not query the origin city
                                unless it is relevant to the user's request.

                                When analyzing the results, highlight travel-relevant conditions such as:
                                - temperature
                                - rain or storms
                                - unusually hot or cold weather
                                - wind
                                - conditions that may affect outdoor activities

                                Do not invent weather information.
                                Base your response only on data returned by the weather tools.

                                Return a concise weather summary that can be used by the itinerary agent
                                to adjust activities, timing, and packing recommendations.
                                """
                                ),
                HumanMessage(content=query)
            ])

            return {
                "messages": [response],
                "weather_started": True,
                "llm_calls": state.get("llm_calls", 0) + 1
            }   
        
        response = await weather_llm.ainvoke(
            state["messages"]
        )
        

        result = {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

        if not response.tool_calls:
            result["weather_results"] = response.content

        return result


    async def iternary_node(state: IternaryState):

        prompt = f"""
        You are a travel iternary planner.

        User Query : {state["user_query"]}

        Flight Results : {state['flight_results']}

        Hotel Results : {state['hotel_results']}

        Weather Results : {state['weather_results']}

        Now create a simple budget-aware iternary simple and easy to follow
        """

        response = await llm.ainvoke([

            SystemMessage(
                content="""
                        Use the flight, hotel, and weather results provided in state.

                        Flight rules:
                        - Do not invent return flights, prices, durations, or schedules.
                        - Do not choose one flight as the user's flight unless the tool results
                        clearly support that choice.
                        - Present available flight options when multiple options are returned.

                        Weather rules:
                        - Use forecast conditions to influence the itinerary.
                        - Prefer indoor activities during rain or poor weather.
                        - Prefer outdoor activities during clearer periods.
                        - Include useful packing or timing recommendations when supported
                        by the weather data.

                        Only use factual details that are present in the agent results.
                        """
            ),

            HumanMessage(prompt)

        ])

        return {

            "iternary": response.content,

            "messages": [
                AIMessage(content=response.content)
            ],

            "llm_calls": state.get("llm_calls", 0) + 1

        }


    async def final_agent(state: IternaryState):

        final_prompt = f"""
        You are a travel iternary planner.

        User Query : {state["user_query"]}

        Flight Results : {state['flight_results']}

        Hotel Results : {state['hotel_results']}

        Itinerary : {state['iternary']}

        Now create a final response to the user
        """

        response = await llm.ainvoke([

            SystemMessage(
                content="You are a travel iternary planner. Create a final response to the user."
            ),

            HumanMessage(final_prompt)

        ])

        return {

            "messages": [
                AIMessage(content=response.content)
            ],

            "llm_calls": state.get("llm_calls", 0) + 1

        }


    builder = StateGraph(IternaryState)

    builder.add_node(
        "flight",
        flight_node
    )

    builder.add_node(
        "flight_tools",
        flight_tool_node
    )

    builder.add_node(
        "tavily_tools",
        tavily_tool_node
    )
    
    builder.add_node(
        "weather_tools",
        weather_tool_node
    )

    builder.add_node(
        "hotel",
        hotel_node
    )

    builder.add_node(
        "weather",
        weather_node
    )

    builder.add_node(
        "iternary",
        iternary_node
    )

    builder.add_node(
        "final",
        final_agent
    )


    builder.add_edge(

        START,

        "flight"

    )


    builder.add_conditional_edges(

        "flight",

        tools_condition,

        {

            "tools": "flight_tools",

            "__end__": "hotel"

        }

    )


    builder.add_conditional_edges(

        "hotel",

        tools_condition,

        {

            "tools": "tavily_tools",

            "__end__": "weather"

        }

    )

    builder.add_conditional_edges(
        "weather",
        tools_condition,
        {
            "tools": "weather_tools",
            "__end__": "iternary"
        }
    )



    builder.add_edge(

        "flight_tools",

        "flight"

    )


    builder.add_edge(

        "tavily_tools",

        "hotel"

    )
    
    builder.add_edge(
        "weather_tools",
        "weather"
    )


    builder.add_edge(

        "iternary",

        "final"

    )


    builder.add_edge(

        "final",

        END

    )


    database_url = get_database_url()



    async with AsyncPostgresSaver.from_conn_string(
        database_url
    ) as checkpointer:

        await checkpointer.setup()

        app = builder.compile(
            checkpointer=checkpointer
        )


        config = {

            "configurable": {

                "thread_id": "trip-8"

            }

        }


        async for chunk in app.astream(
            {
            "user_query": "I want to go delhi to Paris for a weekend trip"
            },
            config=config,
            stream_mode="updates"
        ):
            print(chunk)


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )