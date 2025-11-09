#!/usr/bin/env python3
"""Test script to verify LLM is working correctly"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.base import BaseAgent
from app.models.schemas import AgentType, AgentStatus

async def test_llm():
    """Test if LLM is working"""
    print("🧪 Testing LLM Connection...")
    print(f"API Key: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET'}")
    print(f"Model: {os.getenv('OPENROUTER_MODEL', 'NOT SET')}")
    print()
    
    try:
        # Create a test agent
        print("Creating test agent...")
        agent = BaseAgent(
            agent_type=AgentType.PLANNER,
            system_prompt="You are a helpful assistant. Respond with JSON format.",
            temperature=0.7
        )
        print("✅ Agent created successfully")
        print()
        
        # Test simple query
        print("Testing simple query...")
        test_input = "What is 2+2? Respond with a JSON object: {\"answer\": number}"
        print(f"Input: {test_input}")
        print()
        
        result = await agent.process(test_input)
        
        print("Result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Response: {result.get('response', 'N/A')[:200]}...")
        print()
        
        if result.get("status") == "success":
            print("✅ LLM is working correctly!")
            return True
        else:
            print(f"❌ LLM returned error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_planner_agent():
    """Test planner agent specifically"""
    print("\n" + "="*50)
    print("🧪 Testing Planner Agent...")
    print("="*50)
    print()
    
    try:
        from app.agents.planner import PlannerAgent
        
        planner = PlannerAgent()
        print("✅ Planner agent created")
        print()
        
        test_goal = "Navigate to (5,6) and collect rock sample"
        print(f"Test goal: {test_goal}")
        print()
        
        steps = await planner.plan_mission(test_goal)
        
        print(f"✅ Generated {len(steps)} steps:")
        for step in steps:
            print(f"  Step {step.step_number}: {step.action}")
            if step.target_position:
                print(f"    Target: ({step.target_position.x}, {step.target_position.y})")
            print(f"    Description: {step.description}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing planner: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rover_agent():
    """Test rover agent specifically"""
    print("\n" + "="*50)
    print("🧪 Testing Rover Agent...")
    print("="*50)
    print()
    
    try:
        from app.agents.rover import RoverAgent
        from app.models.schemas import MissionStep, RoverPosition
        
        rover = RoverAgent()
        print("✅ Rover agent created")
        print()
        
        test_step = MissionStep(
            step_number=1,
            action="collect",
            target_position=RoverPosition(x=5, y=6),
            description="Collect rock sample at (5,6)",
            completed=False
        )
        
        current_pos = RoverPosition(x=5, y=6)
        obstacles = []
        mission_goal = "Navigate to (5,6) and collect rock sample"
        
        print(f"Test step: {test_step.action}")
        print(f"Current position: ({current_pos.x}, {current_pos.y})")
        print(f"Mission goal: {mission_goal}")
        print()
        
        result = await rover.execute_step(test_step, current_pos, obstacles, mission_goal=mission_goal)
        
        print("✅ Rover agent response:")
        print(f"  Status: {result.get('status')}")
        print(f"  Action: {result.get('action')}")
        print(f"  Findings: {result.get('findings', 'N/A')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        print()
        
        if result.get("status") == "success" and result.get("findings"):
            print("✅ Rover agent generated findings correctly!")
            return True
        else:
            print("⚠️  Rover agent response missing findings")
            return False
        
    except Exception as e:
        print(f"❌ Error testing rover: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("="*50)
    print("LLM Functionality Test Suite")
    print("="*50)
    print()
    
    # Test 1: Basic LLM connection
    test1 = await test_llm()
    
    # Test 2: Planner agent
    test2 = await test_planner_agent()
    
    # Test 3: Rover agent with findings
    test3 = await test_rover_agent()
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Basic LLM: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Planner Agent: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Rover Agent: {'✅ PASS' if test3 else '❌ FAIL'}")
    print()
    
    if all([test1, test2, test3]):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

