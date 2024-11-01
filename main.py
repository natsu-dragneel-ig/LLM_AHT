import csv
import json
import time
from simulation.unity_simulator import comm_unity
import utils
import sklweka.jvm as jvm
import pandas as pd
import numpy as np

# initiate the simluator
comm = comm_unity.UnityCommunication(port='8080')

jvm.start()

env_id = 6

for iteration_ in range(10):
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
    fridge_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fridge'][0]
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
        'fridge': fridge_id,
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

    # Select an initial state
    initialscript, human_act, ah_act, weekday = utils.select_initialstate(id_dict, '<char0>', '<char1>') # , all_rows[iteration_]
    # Prepare domain
    for script_instruction in initialscript:
        act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        print(act_success, human_success, ah_success, message)
        if not act_success or not human_success or not ah_success:
            exit(1)
        success, graph = comm.environment_graph()

    if not act_success or not human_success or not ah_success:
        exit(1)

    step = 0
    task_counter = 0
    prev_human_actions = ['None', 'None']
    prev_ah_actions = ['None', 'None']
    prev_human_tasks = ['None', 'None']
    prev_ah_tasks = ['None', 'None']
    completed_tasks = []
    goal = False
    human_success = False
    ah_success = False
    current_script = initialscript # since the ids do not change
    old_human_actions = []
    old_ah_actions = []

    human_at_dict = {
        'kitchentable': 0,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0,
        'kitchen': 0
    }

    ah_at_dict = {
        'kitchentable': 0,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0,
        'kitchen': 0
    }

    # ---------------------------------- START: AD HOC TEAMWORK ---------------------------------- #
    # get the tasks for the human
    all_tasks, flags = utils.get_human_tasks(weekday)
    total_tasks = len(all_tasks)
    ah_sub_goal_success = True # for the first time step
    human_sub_goal_success = False
    last_task = False
    fail_count = 0
    ah_tasks_count = 0
    # main goal loop
    human_tasks = all_tasks.copy()
    ah_agent_tasks = all_tasks.copy()
    game_time = 0
    plan_time = 0
    llm_task_list = []
    while human_tasks:
        sub_goal = human_tasks.pop(0)
        human_sub_goal_success = False
        while (not human_sub_goal_success):
            if ah_sub_goal_success:
                if ah_agent_tasks:
                    ah_task = ah_agent_tasks.pop(0)
                else:
                    ah_task = sub_goal
                current_task = [utils.map_ASP_goal_LLM_text(ah_task)]
                if ah_tasks_count < total_tasks and len(completed_tasks) < total_tasks:
                    if not llm_task_list:
                        llm_task_list = utils.get_llm_next_task(flags, completed_tasks)
                        print(llm_task_list)
                        llm_task_list = utils.get_ordered_tasks(flags, llm_task_list.copy())
                        print(llm_task_list)
                else:
                    last_task = True
                if len(llm_task_list) > 1:
                    if llm_task_list[0] == current_task[0]:
                        future_task = llm_task_list.pop(0)
                        future_task = llm_task_list.pop(0)
                        current_task = [current_task[0],future_task]
                        ah_tasks_count += 2
                        ah_ASP_goal = utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(current_task[0])) + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(current_task[0]))) + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                    else:
                        future_task = llm_task_list.pop(0)
                        current_task = [current_task[0],future_task]
                        ah_tasks_count += 1
                        ah_ASP_goal = utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(current_task[0])) + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(current_task[0]))) + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                elif llm_task_list:
                    if llm_task_list[0] == current_task[0]:
                        future_task = llm_task_list.pop(0)
                        ah_tasks_count += 1
                        ah_ASP_goal = utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(current_task[0]))
                        ah_const_timeout = np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(current_task[0])))
                    else:
                        future_task = llm_task_list.pop(0)
                        ah_tasks_count += 2
                        current_task = [current_task[0],future_task]
                        ah_ASP_goal = utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(current_task[0])) + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(current_task[0]))) + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                else:
                    ah_tasks_count += 1
                    ah_ASP_goal = utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(current_task[0]))
                    ah_const_timeout = np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(current_task[0])))
            
            human_actions, ah_fluents, common_fluents, human_sub_goal_success = utils.run_ASP_human(graph, id_dict, sub_goal, human_at_dict, ah_at_dict)

            if human_actions is None:
                print('cannot find a satisifying solution within the given const n value.')
                human_actions = old_human_actions.copy()

            # check whether sub goal is achived
            if human_sub_goal_success:
                task_counter = task_counter+1
                prev_human_tasks.pop(0)
                prev_human_tasks.append(sub_goal)
                completed_tasks = utils.add_unique([utils.map_ASP_goal_LLM_text(sub_goal)],completed_tasks)
                continue
            
            ah_actions, ah_sub_goal_success, plan_time_inside = utils.run_ASP_ahagent(graph, sub_goal, ah_ASP_goal, ah_const_timeout, prev_human_tasks, flags, ah_fluents, common_fluents, prev_human_actions.copy(), prev_ah_actions.copy(), env_id, id_dict, current_script.copy(), last_task)
            plan_time = plan_time+plan_time_inside

            if ah_actions is None:
                print('cannot find a satisifying solution within the given timeoutvalue.')
                if fail_count > 0 or last_task:
                    ah_actions = []
                else:
                    ah_actions = old_ah_actions.copy()
            if ah_sub_goal_success:
                for item in current_task:
                    if utils.map_LLM_text_ASP_goal(item) in all_tasks:
                        completed_tasks = utils.add_unique([item],completed_tasks)
                if not last_task:
                    continue
            
            script = utils.generate_script(human_actions.copy(), ah_actions.copy(), id_dict, '<char0>', '<char1>')
            if not script:
                if fail_count > 0:
                    break
                continue
            current_script.append(script[0])

            # save data for model update
            values = utils.save_model_data(graph, sub_goal, prev_human_tasks.copy(), script.copy(), prev_human_actions.copy(), flags, id_dict)
            if values:
                with open('state_new.csv', 'a', newline='') as f:  
                    writer = csv.writer(f)
                    writer.writerow(values)
        
            if '|' in script[0]:
                script_split = script[0].split('|')
                act1 = script_split[0]
                act2 = script_split[1]
                act1_split = (act1.split(' '))[1:]
                act2_split = (act2.split(' '))[1:]
                if act1_split == act2_split and act1_split[0] == '[grab]': # both agents grab same item
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(script_split[0])
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append(act1)
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(act2)
            else:
                # only ad hoc agent is not countered here
                prev_human_actions.pop(0)
                prev_human_actions.append(script[0])
                prev_ah_actions.pop(0)
                prev_ah_actions.append('None')

            # worksround to overcome virtualhome random behaviour issues
            if '|' in script[0]:
                script_split = script[0].split('|')
                human_act = script_split[0].replace('char0','char1')
                ah_act = script_split[1]
                if human_act == ah_act and 'grab' in human_act:
                    script[0] = ah_act

            game_time_start = time.time()
            act_success, human_success, ah_success, message = comm.render_script([script[0]], recording=False, skip_animation=True)
            game_time_end = time.time()
            game_time = game_time + (game_time_end-game_time_start)

            if not act_success:
                fail_count += 1
                if human_success and not ah_success:
                    if '|' in script[0]:
                        script_split = script[0].split('|')
                        human_act = script_split[0]
                        ah_act = None
                    else:
                        human_act = script[0]
                        ah_act = None
                    if human_actions and human_act and human_success and ('move' in human_actions[0]):
                        human_action_split = (human_act.split(' '))[1:]
                        location = utils.get_location(human_action_split[2][1:-1], id_dict)
                        for key, value in human_at_dict.items():
                            if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                                human_at_dict[key] = 1
                            else:
                                human_at_dict[key] = 0
                else:
                    if fail_count > 5:
                        break
            else:
                if '|' in script[0]:
                    script_split = script[0].split('|')
                    human_act = script_split[0]
                    ah_act = script_split[1]
                else:
                    human_act = script[0]
                    ah_act = None
                if human_actions and human_act and human_success and ('move' in human_actions[0]):
                    human_action_split = (human_act.split(' '))[1:]
                    location = utils.get_location(human_action_split[2][1:-1], id_dict)
                    for key, value in human_at_dict.items():
                        if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                            human_at_dict[key] = 1
                        else:
                            human_at_dict[key] = 0
                if ah_act and ah_success and ('move' in ah_actions[0]):
                    ah_action_split = (ah_act.split(' '))[1:]
                    location = utils.get_location(ah_action_split[2][1:-1], id_dict)
                    for key, value in ah_at_dict.items():
                        if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                            ah_at_dict[key] = 1
                        else:
                            ah_at_dict[key] = 0
            # get the graph
            success, graph = comm.environment_graph()
            # Previous action of the agent (include multiple actions in their particular order?)
            step = step+1
            if human_actions and len(human_actions) > 1:
                old_human_actions = human_actions[1:]
            if ah_actions and len(ah_actions) > 1:
                old_ah_actions = ah_actions[1:]

    # utils.write_predict_action(llm_file, ','.join(map(str,completed_tasks)))
    print(step, task_counter, game_time, plan_time)
jvm.stop()
