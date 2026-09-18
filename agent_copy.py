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



##new copy

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


from pydantic import BaseModel, Field

from langgraph.types import interrupt, Command
from langgraph.types import Send

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
    flight_messages: Annotated[List[AnyMessage], operator.add]
    hotel_messages: Annotated[List[AnyMessage], operator.add]
    budget_messages: Annotated[List[AnyMessage], operator.add]
    weather_messages: Annotated[List[AnyMessage], operator.add]

    user_query: str

    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str

    iternary: str

    llm_calls: Annotated[int, operator.add]

    gaurdrail_allowed: bool
    gaurdrail_reason: str

    selected_agents: List[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str


    ##HITL
    approval: bool
    approval_request: str
    human_feedback: str

    flight_started: bool
    hotel_started: bool

    final_response: str


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

    flight_tool_node = ToolNode(flight_tools, messages_key="flight_messages")

    tavily_llm = llm.bind_tools(tavily_tools)

    tavily_tool_node = ToolNode(tavily_tools, messages_key="hotel_messages")

    weather_llm = llm.bind_tools(weather_tools)

    weather_tool_node = ToolNode(weather_tools, messages_key="weather_messages")

    KNOWN_AGENTS = ["flight", "hotel", "weather", "budget"]

    AGENT_ORDER = ["flight", "hotel", "weather", "budget"]

    trip_constraints = {
        "destination": "",
        "origin": "",
        "duration": 0,
        "budget": 0,
        "travel_style": "",
        "special_requests": ""
    }

    class gaurdrail_agent_state(BaseModel):
        allowed: bool
        reason: str
        details_not_enough: bool

    llm_gaurdrail = llm.with_structured_output(gaurdrail_agent_state)

    llm_supervisor = llm.with_structured_output(SupervisorState)

    class SupervisorState(BaseModel):
        destination: str | None = Field(
            default=None,
            description="The destination city for the trip, if provided"
        )

        origin: str | None = Field(
            default="Delhi",
            description="The origin city for the trip, if provided"
        )

        duration: int | None = Field(
            default=None,
            description="The duration of the trip in days, if provided"
        )

        budget: int | None = Field(
            default=None,
            description="The budget for the trip, if provided"
        )

        travel_style: str | None = Field(
            default=None,
            description="The travel style, if provided (e.g., budget, luxury, adventure)"
        )

        special_requests: str | None = Field(
            default=None,
            description="Any special requests or preferences, if provided"
        )

        selected_agents: List[str] = Field(
            default=[],
            description="The agents selected for the trip, if provided"
        )

        reasoning: str = Field(
            default="",
            description="The reasoning for the supervisor's decision of selecting the particular agents"
        )

    async def gaurdrail_agent(state: IternaryState):
        GUARDRAIL_SYSTEM_PROMPT = """
        You are a guardrail agent for a travel-planning system.

        Check two things:
        1. Whether the request is safe.
        2. Whether a clear travel destination is provided.

        Set `details_not_enough=True` if the destination is missing or unclear.
        Origin, duration, budget, travel style, and special requests are optional.
        state the reason as destination is required

        Set `allowed=False` for harmful, illegal, dangerous, exploitative, or criminal requests 
        and set `allowed=False` if the details_not_enough is true

        If the request is unsafe:
        - allowed = False
        - details_not_enough = False

        If the request is safe but destination is missing:
        - allowed = False
        - details_not_enough = True

        If the request is safe and destination is clear:
        - allowed = True
        - details_not_enough = False

        Keep `reason` short and concise.
        Do not invent missing details.
        """
        query = state["user_query"]

        gaurdrail_prompt = f"""
        You are a gaurdrail agent. 
        User Query : {query}
        Now check if the query is valid or not
        """

        response = await llm_gaurdrail.ainvoke([
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            HumanMessage(gaurdrail_prompt)
        ])

        print(response)

        result = {
            "llm_calls": 1,
            "gaurdrail_allowed": response.allowed,
            "gaurdrail_reason": response.reason,
            "details_not_enough": response.details_not_enough,
            "messages":[AIMessage(content="Gaurdrail agent has checked the query, good to proceed")]
        }

        if not response.allowed:
            result["messages"] = [
                AIMessage(content=response.reason)
            ]
            result["final_response"] = response.reason
        
        return result
    
    async def guardrail_router(state: IternaryState):

        if not state["gaurdrail_allowed"]:
            return "end"

        return "supervisor"
    
    async def supervisor_node(state: IternaryState):

        supervisor_prompt = f"""
        You are the supervisor of a multi-agent travel planning system.

        Available agents:
        {KNOWN_AGENTS}

        Original user request:
        {state["user_query"]}

        Current trip constraints:
        {state.get("trip_constraints", {})}

        Select the agents required and extract the current trip details.
        """

        if state.get("human_feedback"):
            supervisor_prompt += f"""

            IMPORTANT: The user rejected the previous itinerary.

            User correction:
            {state["human_feedback"]}

            The user's latest feedback OVERRIDES any conflicting information
            from the original request or previous trip constraints.

            For example:
            - If original destination was Paris
            - and feedback says destination should be Dubai
            - destination MUST now be Dubai.

            Update the extracted trip constraints according to the latest feedback.

            Select only the agents that need to rerun because of this correction.
            """
        
        response = await llm_supervisor.ainvoke([
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content="Determine the CURRENT travel plan.")
        ])


        requested_agents = response.selected_agents

        selected_agents = [agent for agent in AGENT_ORDER if agent in requested_agents]

        trip_constraints["destination"] = response.destination
        trip_constraints["origin"] = response.origin
        trip_constraints["duration"] = response.duration
        trip_constraints["budget"] = response.budget
        trip_constraints["travel_style"] = response.travel_style
        trip_constraints["special_requests"] = response.special_requests

        print("trip_constraints", trip_constraints)

        return {
            "selected_agents": selected_agents,
            "supervisor_reasoning": response.reasoning,
            "trip_constraints": trip_constraints,
            "llm_calls":  1,
            "messages": [AIMessage(content="Supervisor has selected the agents")]
        }

    def supervisor_router(state):
        return [
            Send(agent, state)
            for agent in state["selected_agents"]
        ]

    async def flight_node(state: IternaryState):
        destination = state["trip_constraints"]["destination"]
        query = f"best flights to {destination}"

        if not state.get("flight_started", False):
            messages = [
                SystemMessage(
                    content="""
                    You are the flight-search agent.

                    When the user asks for travel between locations,
                    use the search_flights tool.

                    Infer the appropriate IATA airport codes yourself.

                    Do not ask for travel dates because the available
                    flight tool does not require dates.
                    """
                ),
                HumanMessage(content=query)
            ]
            

        else:
            messages = [*state["flight_messages"]]
        
            if state.get("human_feedback"):
                messages = [*state["flight_messages"]]
                if (state.get("human_feedback") and not isinstance(messages[-1], ToolMessage)):
                    messages.append(
                    HumanMessage(
                        content=f"""
                        The user rejected the previous plan.

                        Current trip constraints:
                        {state["trip_constraints"]}

                        Feedback:
                        {state["human_feedback"]}

                        Update your flight research according to this feedback.
                        """
                    )
                )

        response = await flight_llm.ainvoke(
            messages
        )

        result = {
            "flight_messages": [response],
            "llm_calls": 1,
            "flight_started": True
        }
        
        if not response.tool_calls:
            result["flight_results"] = response.content

        return result



    async def hotel_node(state: IternaryState):
        destination = state["trip_constraints"]["destination"]
        query =  f"best hotels in {destination}"

        if not state.get("hotel_started", False):
            messages = [
                SystemMessage(content="You are a hotel search agent. Use the tavily_search tool to find hotels."),
                HumanMessage(content=query)
            ]
        else:
            messages = [*state["hotel_messages"]]

            if (state.get("human_feedback") and not isinstance(messages[-1], ToolMessage)):
                    messages.append(
                        HumanMessage(
                            content=f"""
                            The user rejected the previous plan.

                            Current trip constraints:
                            {state["trip_constraints"]}

                            Feedback:
                            {state["human_feedback"]}

                            Update your hotel research according to this feedback.
                            """
                        )
                    )

        
        response = await tavily_llm.ainvoke(
            messages
        )
        
        result = {
            "hotel_messages": [response],
            "llm_calls":  1,
            "hotel_started": True
        }

        if not response.tool_calls:
            result["hotel_results"] = response.content

        return result

    
    async def budget_node(state: IternaryState):
        prompt = f"""
        You are a budget planning agent. 
        User Query : {state["user_query"]}
        trip_constraints : {state['trip_constraints']}
        
        Human feedback:
        {state.get("human_feedback", "None")}
        
        Now create a budget plan for the user, mention clearly it as estimates not guaranteed
        """

        response = await llm.ainvoke([
            SystemMessage(content="You are a budget planning agent. Create a budget plan for the user."),
            HumanMessage(prompt)
        ])

        return {
            "budget_results": response.content,
            "budget_messages": [AIMessage(content=response.content)],
            "llm_calls": 1
        }

    
    async def weather_node(state: IternaryState):

        destination = state["trip_constraints"]["destination"]
        query = f"Weather in {destination}"

        # First run
        if not state.get("weather_started", False):

            messages = [
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
                    - Focus primarily on the destination city.
                    - Do not query the origin city unless relevant.

                    When analyzing results, highlight travel-relevant conditions such as:
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
            ]

        # Tool continuation or HITL revision
        else:

            messages = [*state["weather_messages"]]

            # Add feedback only when starting a revision,
            # NOT after a ToolMessage.
            if (
                state.get("human_feedback")
                and not isinstance(messages[-1], ToolMessage)
            ):
                messages.append(
                    HumanMessage(
                        content=f"""
                        The user rejected the previous travel plan.

                        Current trip constraints:
                        {state["trip_constraints"]}

                        Feedback:
                        {state["human_feedback"]}

                        Update your weather research according to this feedback.

                        The latest trip constraints and user feedback override
                        conflicting details from the original request.
                        """
                    )
                )

        response = await weather_llm.ainvoke(messages)

        result = {
            "weather_messages": [response],
            "weather_started": True,
            "llm_calls": 1,
        }

        if not response.tool_calls:
            result["weather_results"] = response.content

        return result


    async def iternary_node(state: IternaryState):
        prompt = f"""
        You are a travel itinerary planner.

        Original User Query:
        {state["user_query"]}

        CURRENT Trip Constraints:
        {state["trip_constraints"]}

        Latest Human Feedback:
        {state.get("human_feedback", "None")}

        Flight Results:
        {state.get("flight_results", "Not requested")}

        Hotel Results:
        {state.get("hotel_results", "Not requested")}

        Budget Results:
        {state.get("budget_results", "Not requested")}

        Weather Results:
        {state.get("weather_results", "Not requested")}

        IMPORTANT:
        The current trip constraints and latest human feedback override
        conflicting details in the original query.

        Create the revised itinerary.
        """

        response = await llm.ainvoke([
            SystemMessage(content="You are a travel iternary planner. Create a simple budget-aware iternary simple and easy to follow."),
            HumanMessage(prompt)
        ])  

        approval_request = "please review the iternary and provide feedback, Approve it to create the final response or Reject and provide feedback"
        
        return {
            "iternary": response.content,
            "messages": [AIMessage(content="Draft iternary created"), ],
            "llm_calls": 1,
            "approval_request": approval_request
        }

    async def human_approval_node(state: IternaryState):
        review = interrupt(
            {
                "question": "Do you approve this request?",
                "approval_request": state["approval_request"],
                "selected_agents": state["selected_agents"],
                "supervisor_reasoning": state["supervisor_reasoning"],
                "draft_iternary": state["iternary"],
                "expected_response":{
                    "approved": True,
                    "feedback": "optional feedback"
                }
            }
        )

        approved = bool(review.get("approved", True))
        feedback = str(review.get("feedback", ""))
        
        return {
            "approval": approved,
            "human_feedback": feedback,
            "messages": [AIMessage(content="Human approval received")]
        }
    
    async def review_router(state):

        if state["llm_calls"] >= 20:
            return "final"

        if not state["approval"]:
            return "supervisor"

        return "final"

    async def final_agent(state: IternaryState):

        final_prompt = f"""
        You are the final travel response agent.

        Original User Query:
        {state["user_query"]}

        CURRENT Trip Constraints:
        {state["trip_constraints"]}

        Latest Human Feedback:
        {state.get("human_feedback", "None")}

        Flight Results:
        {state.get("flight_results", "Not requested")}

        Hotel Results:
        {state.get("hotel_results", "Not requested")}

        Budget Results:
        {state.get("budget_results", "Not requested")}

        Approved Itinerary:
        {state["iternary"]}

        IMPORTANT:
        - The CURRENT trip constraints are authoritative.
        - The approved itinerary is authoritative.
        - Latest human feedback overrides conflicting information in the original query.
        - Do NOT revert to details from the original request if they were changed later.
        - Produce the final response based on the latest approved plan.
        """
        
        response = await llm.ainvoke([
            SystemMessage(content="Create the final response using the latest approved travel plan."),
            HumanMessage(content=final_prompt)
        ])
        
        return {
            "messages": [AIMessage(content=response.content)],
            "final_response": response.content,
            "llm_calls": 1
        }
    
    async def flight_router(state):
        last_message = state["flight_messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return "__end__"


    async def hotel_router(state):
        last_message = state["hotel_messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return "__end__"

    builder = StateGraph(IternaryState)

    builder.add_node("gaurdrail", gaurdrail_agent)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("flight", flight_node)
    builder.add_node("budget", budget_node)
    builder.add_node("hotel", hotel_node)
    builder.add_node("iternary", iternary_node, defer=True)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("final", final_agent)

    builder.add_node("tavily_tools", tavily_tool_node)
    builder.add_node("flight_tools", flight_tool_node)

    builder.add_edge(
        START,
        "gaurdrail"
    )

    builder.add_conditional_edges(
        "gaurdrail",
        guardrail_router,
        {
            "supervisor": "supervisor",
            "end": END
        }
    )

    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        ["flight", "hotel", "budget"]
    )

    builder.add_conditional_edges(
        "flight",
        flight_router,
        {
            "tools": "flight_tools",
            "__end__": "iternary"
        }
    )

    builder.add_conditional_edges(
        "hotel",
        hotel_router,
        {
            "tools": "tavily_tools",
            "__end__": "iternary"
        }
    )

    builder.add_conditional_edges(
        "human_approval",
        review_router,
        {
            "supervisor": "supervisor",
            "final": "final"
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
        "budget",
        "iternary"
    )

    builder.add_edge(
        "iternary",
        "human_approval"
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

                "thread_id": "trip-10"

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


