import csv
from simulation.unity_simulator import comm_unity
import utils
import utils_partial
import time
import sklweka.jvm as jvm
import numpy as np
from copy import deepcopy

# initiate the simluator
comm = comm_unity.UnityCommunication(port='8080')

jvm.start()

# select the environment
env_id = 6

tasks = [
    ['Breakfast_weekday', 'Workstation', 'Coffee', 'Serve_snacks']
]

all_flags = [
{'weekday': True, 'office': False, 'lunch': False}
]

for iteration_ in range(1):
    # clean action file
    open('tobe_explored_actions.txt', 'w').close()

    comm.reset(env_id)
    # Get the state observation
    success, graph = comm.environment_graph()

    # remove unnecessary objects and prepare the domain
    success1, message, success2, graph = utils.clean_graph(comm, graph, ['chicken'])

    # Get nodes for differnt objects
    cereal_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cereal'][0]
    breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
    banana_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bananas'][0]
    plum_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'plum'][0]
    cupcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cupcake'][0]
    cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
    chips_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'chips'][0]
    candybar_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'candybar'][0]
    milk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'milk'][0]
    wine_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'wine'][0]
    juice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'juice'][0]
    cellphone_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cellphone'][1]
    dishwasher_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
    computer_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'computer'][0]
    coffeemaker_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeemaker'][0]
    plate_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'plate'][0]
    waterglass_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][1]
    mug_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'mug'][0]
    coffeepot_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeepot'][0]
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    livingcoffeetable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeetable'][0] # livingroom
    livingdesk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'desk'][0] # livingroom
    bedcoffeetable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeetable'][1] # bedroom
    beddesk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'desk'][2] # bedroom - two desks here
    book_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'book'][0]
    boardgame_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'boardgame'][0]
    tvstand_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'tvstand'][0]
    counterone_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0] # counter_one
    countertwo_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0] # counter_two
    counterthree_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0] # counter_thre
    apple_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'apple'][11]
    chair_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'chair'][2]
    bookshelf_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bookshelf'][2]
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    sofa_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'sofa'][0]

    # collect and process data
    id_dict = {
        'cereal': cereal_id,
        'breadslice': breadslice_id,
        'bananas': banana_id,
        'plum': plum_id,
        'cupcake': cupcake_id,
        'cutlets': cutlets_id,
        'chips': chips_id,
        'candybar': candybar_id,
        'milk': milk_id,
        'wine': wine_id,
        'juice': juice_id,
        'dishwasher': dishwasher_id,
        'computer': computer_id,
        'coffeemaker': coffeemaker_id,
        'plate': plate_id,
        'waterglass': waterglass_id,
        'mug': mug_id,
        'coffeepot': coffeepot_id,
        'kitchentable': kitchentable_id,
        'livingroom_coffeetable': livingcoffeetable_id,
        'livingroom_desk': livingdesk_id,
        'bedroom_coffeetable': bedcoffeetable_id,
        'bedroom_desk': beddesk_id,
        'book': book_id,
        'boardgame': boardgame_id,
        'kitchen_smalltable': tvstand_id,
        'counter_one': counterone_id,
        'stove': countertwo_id,
        'counter_three': counterthree_id,
        'cellphone': cellphone_id,
        'apple': apple_id,
        'chair': chair_id,
        'bookshelf': bookshelf_id,
        'microwave': microwave_id,
        'sofa': sofa_id,
        'floor': 204
    }

    # Add human
    comm.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Male1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Female2', initial_room='kitchen')

    num_agents = 3

    # Select an initial state
    initialscript, _, _, weekday = utils.select_initialstate(id_dict, '<char0>', '<char1>', '<char2>') # , all_rows[iteration_]
    # Prepare domain
    for script_instruction in initialscript:
        act_success, success_msgs, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        human_success = success_msgs[0]
        ah1_success = success_msgs[1]
        ah2_success = success_msgs[2]
        # only the first two agents are performing the actions in the initialscript
        if not act_success or not human_success or not ah1_success:
            exit(1)
        success, graph = comm.environment_graph()

    if not act_success or not human_success or not ah1_success:
        exit(1)
    # dependency in virtualhome
    human_at_dict = {
        'kitchentable': 1,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0
    }
    ah1_at_dict = {
        'kitchentable': 1,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0
    }
    ah2_at_dict = {
        'kitchentable': 1,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0
    }

    # virtualhome issue fix - completely tired of fixing random issues in env
    status_switchon_dict = {
        'dishwasher': 0,
        'computer': 0,
        'coffeemaker': 0
    }

    # ---------------------------------- START: AD HOC TEAMWORK ---------------------------------- #
    step = 0
    task_counter = 0
    prev_actions = []
    for agent in range(num_agents):
        prev_actions.append(['None', 'None'])
    prev_tasks = []
    for agent in range(num_agents):
        prev_tasks.append(['None', 'None'])
    current_script = initialscript # since the ids do not change

    all_tasks = tasks[iteration_]
    flags = all_flags[iteration_]
    print(all_tasks, flags)
    # setting last task for the human for using the complete goal
    human_last_task = False
    ah1_last_task = False
    ah2_last_task = False

    # agent number dependent parameters
    # define success parameter for each of the agentsin the env
    human_success = False
    ah1_success = False
    ah2_success = False
    # for storing old actions that can be used in a unsuccessful planning
    old_human_actions = []
    old_ah1_actions = []
    old_ah2_actions = []
    # sucess of the last task
    ah1_sub_goal_success = True # for the first time step
    ah2_sub_goal_success = True # for the first time step
    # oracle tasks for the agents
    human_tasks = deepcopy(all_tasks)
    ah1_agent_tasks = deepcopy(all_tasks)
    ah2_agent_tasks = deepcopy(all_tasks)
    # fail count for ad hoc agent plan fails
    fail_count = 0
    failed_action = False
    total_tasks = len(all_tasks)

    while human_tasks:
        if len(human_tasks) == 1:
            human_last_task = True
        sub_goal = human_tasks.pop(0)
        human_sub_goal_success = False
        while (not human_sub_goal_success):
            # ad hoc agent 1
            if ah1_sub_goal_success:
                if len(ah1_agent_tasks) == 1:
                    ah1_last_task = True
                if ah1_agent_tasks:
                    ah1_current_task = ah1_agent_tasks.pop(0)
                else:
                    ah1_current_task = sub_goal
                ah1_ASP_goal = utils.map_goal_ASP(ah1_current_task)
                ah1_const_timeout = np.array(utils.get_const_timeout(ah1_current_task))

            # ah hoc agent 2 - axiom learning agent!
            if ah2_sub_goal_success:
                if len(ah2_agent_tasks) == 1:
                    ah2_last_task = True
                if ah2_agent_tasks:
                    ah2_current_task = ah2_agent_tasks.pop(0)
                else:
                    ah2_current_task = sub_goal
                ah2_ASP_goal = utils.map_goal_ASP(ah2_current_task)
                ah2_const_timeout = np.array(utils.get_const_timeout(ah2_current_task))

            human_actions, ah1_fluents, ah2_fluents, common_fluents, human_sub_goal_success = utils.run_ASP_human(graph, id_dict, sub_goal, human_at_dict, ah1_at_dict, ah2_at_dict, status_switchon_dict, False)
            if human_actions is None:
                print('cannot find a satisifying solution within the given const n value.')
                human_actions = deepcopy(old_human_actions)

            # check whether sub goal is achived
            if human_sub_goal_success:
                task_counter = task_counter+1
                prev_tasks[0].pop(0)
                prev_tasks[0].append(sub_goal)
                continue
            
            # add ah agent
            ah1_actions, ah1_sub_goal_success = utils.run_ASP_ahagent(graph, sub_goal, ah1_ASP_goal, ah1_const_timeout, deepcopy(prev_tasks), flags, ah1_fluents, common_fluents, deepcopy(prev_actions), env_id, id_dict, deepcopy(current_script), None, False, num_agents, 1)

            comm_action, comm_exec = utils_partial.get_human_interpretation(graph, ah1_actions[0] if ah1_actions else None)

            # add other agent - axiom learning agent!
            ah2_actions, ah2_sub_goal_success, _ = utils_partial.run_ASP_ahagent(graph, sub_goal, ah2_ASP_goal, ah2_const_timeout, deepcopy(prev_tasks), flags, ah2_fluents, common_fluents, deepcopy(prev_actions), env_id, id_dict, deepcopy(current_script), comm_action, comm_exec, False, num_agents, 2, ah2_at_dict, status_switchon_dict)

            if ah1_actions is None:
                print('cannot find a satisifying solution within the given timeoutvalue.')
                if fail_count > 0 or ah1_last_task:
                    ah1_actions = []
                else:
                    ah1_actions = deepcopy(old_ah1_actions)

            if ah1_sub_goal_success and not ah1_last_task:
                continue

            if ah2_actions is None: # nned to be edited here!
                print('cannot find a satisifying solution within the given timeoutvalue.')
                if fail_count > 0 or ah2_last_task:
                    ah2_actions = []
                else:
                    ah2_actions = deepcopy(old_ah2_actions)
            elif ah2_actions == 'Failed':
                ah2_actions = None
                failed_action = True

            if ah2_sub_goal_success and not ah2_last_task:
                continue
            
            script = utils.generate_script(human_actions[0], ah1_actions[0] if ah1_actions else None, ah2_actions[0] if ah2_actions else None, id_dict, '<char0>', '<char1>', '<char2>')

            if not script:
                if fail_count > 0:
                    break
                continue

            if '|' in script[0]:
                script_split = script[0].split('|')
                action_map = {f'<char{i}>': f'act{i+1}' for i in range(num_agents)}  # 4 characters
                action_vars = {f'act{i+1}': None for i in range(num_agents)}  # all actions = None

                # Assign actions
                for part in script_split:
                    for char, action in action_map.items():
                        if char in part:
                            action_vars[action] = part  # Assign the string part to the corresponding action variable
                            break
                act1, act2, act3 = action_vars.values()
            else:
                # single action (no '|')
                action_map = {f'<char{i}>': f'act{i+1}' for i in range(num_agents)}  # 4 characters
                action_vars = {f'act{i+1}': None for i in range(num_agents)}  # all actions = None

                for char, action in action_map.items():
                    if char in script[0]:
                        action_vars[action] = script[0]
                        break
                act1, act2, act3 = action_vars.values()

            act1_split = (act1.split(' '))[1:] if act1 else None # ['[grab]', '<bananas>', '(251)']
            act2_split = (act2.split(' '))[1:] if act2 else None
            act3_split = (act3.split(' '))[1:] if act3 else None

            if act1 and act2 and act3 and act1_split == act2_split == act3_split and ('grab' in act1 or 'switchon' in act1):
                act1 = None
                act1_split = None
                act2 = None
                act2_split = None
                script[0] = act3

            if act1 and act2 and act1_split == act2_split and ('grab' in act1 or 'switchon' in act1):
                act1 = None
                act1_split = None
                script[0] = act2 + (('|' + act3) if act3 else '')

            if act1 and act3 and act1_split == act3_split and ('grab' in act1 or 'switchon' in act1):
                act1 = None
                act1_split = None
                script[0] = ((act2+'|') if act2 else '') + act3

            if act2 and act3 and act2_split == act3_split and ('grab' in act2 or 'switchon' in act2):
                act2 = None
                act2_split = None
                script[0] = ((act1+'|') if act1 else '') + act3

            current_script.append(script[0])
            print(script[0])

            if act1:
                # for human
                values = utils.process_state_new(graph, sub_goal, prev_tasks[0].copy(), prev_actions[0].copy(), flags, id_dict, 0, num_agents, act1)
                if values:
                    with open('state_0.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[0].pop(0)
                prev_actions[0].append(act1)
            else:
                prev_actions[0].pop(0)
                prev_actions[0].append('None')
            if act2:
                # for ad hoc agent1
                values = utils.process_state_new(graph, sub_goal, prev_tasks[1].copy(), prev_actions[1].copy(), flags, id_dict, 1, num_agents, act2)
                if values:
                    with open('state_1.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[1].pop(0)
                prev_actions[1].append(act2)
            else:
                prev_actions[1].pop(0)
                prev_actions[1].append('None')
            if act3:
                # for ad hoc agent2
                values = utils.process_state_new(graph, sub_goal, prev_tasks[2].copy(), prev_actions[2].copy(), flags, id_dict, 2, num_agents, act3)
                if values:
                    with open('state_2.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[2].pop(0)
                prev_actions[2].append(act3)
            else:
                prev_actions[2].pop(0)
                prev_actions[2].append('None')

            act_success, success_msgs, message = comm.render_script([script[0]], recording=False, skip_animation=True)
            human_success = success_msgs[0]
            ah1_success = success_msgs[1]
            ah2_success = success_msgs[2]

            if human_success:
                if human_actions and act1 and ('move' in human_actions[0]):
                    location = utils.get_location(act1_split[2][1:-1], id_dict)
                    for key, value in human_at_dict.items():
                        if key == location:
                            human_at_dict[key] = 1
                        else:
                            human_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1
            if ah1_success:
                if ah1_actions and act2 and ('move' in ah1_actions[0]):
                    location = utils.get_location(act2_split[2][1:-1], id_dict)
                    for key, value in ah1_at_dict.items():
                        if key == location:
                            ah1_at_dict[key] = 1
                        else:
                            ah1_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1
            if ah2_success:
                if ah2_actions and act3 and ('move' in ah2_actions[0]):
                    location = utils.get_location(act3_split[2][1:-1], id_dict)
                    for key, value in ah2_at_dict.items():
                        if key == location:
                            ah2_at_dict[key] = 1
                        else:
                            ah2_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1

            if failed_action: # new action
                failed_action = False
                ah2_sub_goal_success = True
                human_tasks.insert(0, ah2_current_task)
                human_last_task = False
                ah1_agent_tasks.insert(0, ah2_current_task)
                ah1_last_task = False
                if not ah2_success:
                    fail_count -= 1
            
            if fail_count > 6:
                break
            # get the graph
            success, graph = comm.environment_graph()
            # Previous action of the agent (include multiple actions in their particular order?)
            step = step+1
            if human_actions and len(human_actions) > 1:
                old_human_actions = human_actions[1:]
            if ah1_actions and len(ah1_actions) > 1:
                old_ah_actions = ah1_actions[1:]
            if ah2_actions and len(ah2_actions) > 1:
                old_add_actions = ah2_actions[1:]

    print(step, task_counter)
jvm.stop()
