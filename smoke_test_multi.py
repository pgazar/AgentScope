import sys, json, uuid, os
sys.path.insert(0, '/Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope')
os.chdir('/Users/pegahzargarian/projects/MCP-1/workspace/AgenticScope')

from agentscope.config import load_config
from agentscope.intake.intake_agent import IntakeAgent
from agentscope.orchestrator.graph import build_graph

cfg    = load_config()
intake = IntakeAgent().run({
    'agent_type': 'tool_use',
    'turn_type':  'multi',
    'has_gt':     'no',
    'kb_format':  'none',
})
print('active_tools:', intake['active_tools'])
graph = build_graph(intake['active_tools'])

state = {
    'run_id':              str(uuid.uuid4())[:8],
    'agent_folder':        'tests/fake_multi_agent',
    'agent_model':         'claude-haiku-4-5-20251001',
    'eval_inputs':         [
        'Hi! My name is Pegah and I am an AI engineer working on agentic systems.',
        'What are the best frameworks for building AI agents?',
        'Can you remind me what I told you about my job at the start?',
        'Based on my role, which of those frameworks should I prioritize?',
    ],
    'kb_path': None, 'gt_path': None,
    'agent_type': 'tool_use', 'turn_type': 'multi',
    'active_tools': intake['active_tools'],
    'expected_tools': [], 'traces': [],
    'baseline_geval_scores': {},
    'ir_results': None, 'behavior_results': None,
    'geval_results': None, 'cost_results': None,
    'adversarial_results': None, 'synth_results': None,
    'final_report': None, 'config': cfg.model_dump(),
}

result = graph.invoke(state)
geval  = result.get('geval_results', {})
print('\n=== Multi-turn G-Eval scores ===')
print(json.dumps(geval.get('scores', {}), indent=2))
print('\n=== Behavior ===')
beh = result.get('behavior_results', {})
print('ghost_action_rate:', beh.get('ghost_action_rate'))
print('\nSMOKE TEST MULTI-TURN PASSED')
