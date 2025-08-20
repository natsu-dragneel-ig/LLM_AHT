import ast
import csv
from itertools import permutations
import os
import random
import re
import subprocess
import time
import numpy as np
import pandas as pd
from simulation.unity_simulator import utils_viz, comm_unity
from sklweka.classifiers import Classifier
from sklweka.dataset import to_instance
from openai import OpenAI
import joblib
from itertools import product
from copy import deepcopy

client = OpenAI()
delimeters = ['(', ')', ',']
human_asp_pre = 'ASP/human_pre.sp'
human_asp = 'ASP/human.sp'
display_marker = 'display'
models = ['human.model', 'ahagent1.model', 'ahagent2.model']
coffee_ = False # inertial

def process_state(graph, sub_goal, prev_agent_tasks, prev_agent_actions, flags, id_dict, char_id, num_agents):
    state = []
    # Previous action of the agent
    act = prev_agent_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])

    act = prev_agent_actions[1].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])
    
    # Location of the agent
    pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][char_id]
    state.append(pose['position'][0]) # x
    state.append(pose['position'][1]) # y
    state.append(pose['position'][2]) # z
    state.append(pose['rotation'][0]) # x
    state.append(pose['rotation'][1]) # y
    state.append(pose['rotation'][2]) # z
    
    # objects related to the goal
    goal_obj = ','.join(get_goal_obj(sub_goal))
    state.append(goal_obj)

    # get objecs in the ahnd of that agent
    object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == char_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    if len(object_ids) == 2: # obj in both hands
        hand_obj = []
        for item in object_ids: # objects in agents hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    elif len(object_ids) == 1:
        hand_obj = []
        hand_obj.append('None')
        for item in object_ids: # objects in human hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    else:
        state.append('None,None')

    # objects with other agents
    agent_obj = []
    for agent_id in range(num_agents):
        if agent_id != char_id:
            object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
            for item in object_ids: # objects in agents hand
                key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
                agent_obj.append(key_)
    if agent_obj:
        state.append(','.join(agent_obj))#
    else:
        state.append('None')

    # current task
    state.append(sub_goal)
    # previous task
    state.append(prev_agent_tasks[-1])

    # flags
    if flags['weekday']:
        state.append(1)
        if flags['office']:
            state.append(1)
        else:
            state.append(0)
        state.append(0)
    else:
        state.append(0)
        state.append(0)
        if flags['guests']:
            state.append(1)
        else:
            state.append(0)
    return state

def answer_set_finder(expression, answer):
    if not re.search('[\>\<\=\!]',expression):
        expression = re.sub('I\+1', 'I', expression)
        expression = re.sub('\(', '\(', expression)
        expression = re.sub('\)', '\)', expression)
        expression = re.sub('[A-Z][0-9]?', '[a-z0-9_]+(?:\(.+?\))?', expression)
        literal = re.findall("(?<!-)"+expression, answer)
    else:
        literal = [expression]
    return literal

def process_answerlist(answer,sub_goal_success):
    answer_list = answer_set_finder('occurs(A,I)', answer)
    action_list = []
    for i in range(len(answer_list)):
        for element in answer_list:
            if re.search(rf',{i}\)$',element) != None:
                action_list.insert(i, element)
    action_list_res = [item for item in action_list if 'exo' not in item]
    if not action_list_res: # exo action to complete the task; action_list and 
        sub_goal_success = True
    return action_list_res, sub_goal_success

# remove objects
def remove_obj_from_environment(obj, comm, graph):
    ids = [node['id'] for node in graph['nodes']]
    obj_ids = [node['id'] for node in graph['nodes'] if node['class_name'] == obj]
    for obj_id in obj_ids:
        if obj_id in ids:
            edges_to_remove = [edge for edge in graph['edges'] if edge['to_id'] == obj_id or edge['from_id'] == obj_id]
            for edge in edges_to_remove:
                graph['edges'].remove(edge)

            nodes_to_remove = [node for node in graph['nodes'] if node['id'] == obj_id]
            for node in nodes_to_remove:
                graph['nodes'].remove(node)
            success, message = comm.expand_scene(graph)

def clean_graph(comm, graph, objects):
    for obj in objects:
        remove_obj_from_environment(obj, comm, graph)
    utils_viz.clean_graph(graph)
    success1, message = comm.expand_scene(graph)
    success2, graph = comm.environment_graph()
    return success1, message, success2, graph

# weka tree
def predict_next_action(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents):
    values = process_state(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents)
    model, header = Classifier.deserialize(models[modelagent_id]) # 0 - human, 1 - ah_agent1, 2 - ah_agent2, 3 - ah_agent3
    # create new instance
    inst = to_instance(header,values) # Instance.create_instance(values)
    inst.dataset = header
    # make prediction
    try:
        index = model.classify_instance(inst)
        action = header.class_attribute.value(int(index))
    except:
        print('Null pointer exception')
        action = None
    return action

def get_object_locations(fluents):
    # holds(on(cereal,kitchentable),0)
    object_location_dct = {}
    for fluent in fluents:
        if 'holds(on(' in fluent:
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            fir_items = [item.split(',')[0] if ',' in item else item for item in items]
            sec_items = [item.split(',')[1] if ',' in item else item for item in items]
            set_objects = ['cereal', 'breadslice', 'bananas', 'plum', 'apple', 'cupcake', 'cutlets', 'chips', 'candybar', 'milk', 'wine', 'juice', 
                           'plate', 'waterglass', 'mug', 'coffeepot', 'cellphone', 'book', 'boardgame']
            set_locations = ['kitchentable', 'livingroom_coffeetable', 'bedroom_coffeetable', 'livingroom_desk', 'bedroom_desk', 'counter_one', 'counter_three', 'kitchen_smalltable']
            objs = [item for item in fir_items if item in set_objects]
            locs = [item for item in sec_items if item in set_locations]
            if objs and locs:
                object_location_dct[objs[0]] = locs[0]
    return object_location_dct

# block in virtualhome: cannot keep a single env since it is impossible to revert the predicted action execution effects and revert back to original script
def get_future_actions(graph, current_task, all_prev_tasks, flags, ah_fluents, common_fluents, ASP_goal, all_prev_actions, env_id, id_dict, current_script, num_agents, current_agent_id):
    future_actions = [[None,None],[None,None]]
    valid_actions = [[None,None],[None,None]]
    step_exception = [[False,False],[False,False]]
    graphs = []
    actual_predict_time = 0
    for i in range(2): # for 2 steps to the future
        future_action_x = [None, None, None]
        missing_location = False
        actual_predict_time_start = time.time()
        predicted_future_actions = []
        for agent_id in range(num_agents):
            if agent_id != current_agent_id:
                try:
                    predicted_future_actions.append(predict_next_action(graph, current_task, all_prev_tasks[agent_id], all_prev_actions[agent_id], flags, id_dict, agent_id, num_agents))
                except Exception as e:
                    step_exception[i][agent_id] = True
        actual_predict_time_end = time.time()
        actual_predict_time += (actual_predict_time_end-actual_predict_time_start)
        graphs.append(graph)
        # check the validity of the predicted actions
        for idx, pred_action in enumerate(predicted_future_actions): # num_agents-1
            agent_id = idx if idx < current_agent_id else idx+1
            # if the other agent is predicted to move to a specific place not move there?
            if not pred_action:
                valid_actions[i][idx] = False # valid actions = [[True,True], [True,True]]
                future_actions[i][idx] = None # future actions = [[act1,act2], [act4,act5]]
                continue

            # assume ad hoc agent does nothing; only other agent actions added to script
            duplicate_aspname_dict = {
                'coffeetable': ['livingroom_coffeetable', 'bedroom_coffeetable'],
                'desk': ['livingroom_desk', 'bedroom_desk']
            }
            future_action = pred_action.split('_')
            if len(future_action) == 2:
                if future_action[0] == 'find' and future_action[1] == 'coffeetable':
                    if current_task in ['Breakfast_weekday','Pack_bag','Lunch','Coffee']:
                        loc = str(id_dict['livingroom_coffeetable'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    elif current_task in ['Breakfast_weekend']:
                        loc = str(id_dict['bedroom_coffeetable'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    else:
                        missing_location = True
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                elif future_action[0] == 'find' and future_action[1] == 'desk':
                    if current_task in ['Breakfast_weekday','Lunch','Coffee','Breakfast_weekend']:
                        loc = str(id_dict['livingroom_desk'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    elif current_task in ['Serve_snack']:
                        loc = str(id_dict['bedroom_desk'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    else:
                        missing_location = True
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                else:
                    if future_action[0] == 'grab':
                        obj = future_action[1]
                        running_agent_name = 'ahagent'+str(current_agent_id)
                        # in hand of the current agent who is modelling the other agents
                        for j in range(num_agents):
                            if j == current_agent_id: # curent agent who is modelling the other agents
                                running_agent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(grabbed('+running_agent_name+',')]
                                if ('holds(grabbed('+running_agent_name+','+ obj + '),0).' in running_agent_positive_inhand):
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                            elif j == 0: # if huma
                                human_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(human,')]
                                if ('holds(agent_hand(human,'+ obj + '),0).' in human_positive_inhand): # already in humans hand
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                            else: # already in the hand of model agent or another ad hoc agent
                                positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(ahagent'+str(j)+',')]
                                if ('holds(agent_hand(ahagent'+str(j)+','+ obj + '),0).' in positive_inhand):
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                        if valid_actions[i][idx] == False: # not val will set off for None as well
                            continue
                        object_location_dict = get_object_locations(ah_fluents+common_fluents)
                        if obj in object_location_dict:
                            obj_location = object_location_dict[obj]
                            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',ASP_goal)]
                            for item in items:
                                if ',' in item:
                                    fir_item = item.split(',')[0]
                                    sec_item = item.split(',')[1]
                                    if fir_item == obj and sec_item == obj_location: # goal achived; do not change the location of that object
                                        valid_actions[i][idx] = False
                                        future_actions[i][idx] = None
                                        break
                            if valid_actions[i][idx] == False: # not val will set off for None as well
                                continue
                    if future_action[1] == 'tvstand':
                        obj = str(id_dict['kitchen_smalltable'])
                    elif future_action[1] == 'kitchencounter':
                        obj = str(id_dict['counter_one'])
                    else:
                        obj = str(id_dict[future_action[1]])
                    if missing_location: # add a script instruction to also find the place
                        future_action_x[idx] = '<char'+str(agent_id)+'> [find] <' + future_action[1] + '> (' + obj + ')'
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + obj + ')'
            else: # putback or putin
                if future_action[1] in id_dict and future_action[2] in id_dict:
                    # check if the predictive agent is holding the particular object
                    if agent_id == 0:
                        positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(human,')]
                        if not('holds(agent_hand(human,'+ future_action[1] + '),0).' in positive_inhand) and not('holds(agent_hand(human,'+ future_action[1] + '),1).' in positive_inhand):
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                            continue
                    else:
                        positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(ahagent'+str(agent_id)+',')]
                        if not('holds(agent_hand(ahagent'+str(agent_id)+','+ future_action[1] + '),0).' in positive_inhand) and not('holds(agent_hand(ahagent'+str(agent_id)+','+ future_action[1] + '),1).' in positive_inhand):
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                            continue
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[future_action[1]]) + ') <' + future_action[2] + '> (' + str(id_dict[future_action[2]]) + ')'
                elif future_action[2] in id_dict:
                    # loc in, obj not in id_dict
                    objs = duplicate_aspname_dict[future_action[1]]
                    keys = []
                    modelagent_object_ids = [edge['to_id'] for edge in graphs[0]['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
                    if modelagent_object_ids:
                        for modelagent_obj in modelagent_object_ids:
                            key_ = list(id_dict.keys())[list(id_dict.values()).index(modelagent_obj)] # performance >_<
                            if key_ in objs:
                                keys.append(key_)
                        if keys:
                            obj = keys[0]
                            modelagent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand('+ ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',')]
                            if not('holds(agent_hand(' + ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',' + obj + '),0).' in modelagent_positive_inhand) and not('holds(agent_hand(' + ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',' + obj + '),1).' in modelagent_positive_inhand):
                                valid_actions[i][idx] = False
                                future_actions[i][idx] = None
                                continue
                            valid_actions[i][idx] = True
                            future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[obj]) + ') <' + future_action[2] + '> (' + str(id_dict[future_action[2]]) + ')'
                        else:
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                    else:
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                elif future_action[1] in id_dict:#
                    # obj in, loc not in id_dict
                    modelagent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+',')]
                    if not('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+','+ future_action[1] + '),0).' in modelagent_positive_inhand) and not('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+','+ future_action[1] + '),1).' in modelagent_positive_inhand):
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                        continue
                    locs = duplicate_aspname_dict[future_action[2]]
                    loc_proximity = []
                    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][agent_id]
                    for loc in locs:
                        id_ = id_dict[loc]
                        loc_pose = [node['obj_transform'] for node in graph['nodes'] if node['id'] == id_][0]
                        prox_human = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(loc_pose['position']))
                        loc_proximity.append(prox_human)
                    proxmin_idx = loc_proximity.index(min(loc_proximity)) # take the less, if equal the first one
                    loc = locs[proxmin_idx]
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[future_action[1]]) + ') <' + future_action[2] + '> (' + str(id_dict[loc]) + ')'
        
        if any(valid_actions[i]):
            script_instruction = ''
            init_script_instruction = ''
            for agent in range(len(valid_actions[i])): # future actions for that step
                if valid_actions[i][agent] and not step_exception[i][agent]: # future action of that agent for that step
                    if missing_location and future_action_x[agent]:
                        init_script_instruction = init_script_instruction + '|' + future_action_x[agent] if init_script_instruction else future_action_x[agent]
                    script_instruction = script_instruction + '|' + future_actions[i][agent] if script_instruction else future_actions[i][agent]                    
            if init_script_instruction: # pre_req for current_script
                current_script.append(init_script_instruction)
            if script_instruction:
                current_script.append(script_instruction)
            else:
                continue
            # initiate a second env
            comm_dummy = comm_unity.UnityCommunication(port='8082')
            comm_dummy.reset(env_id)
            success_dummy, graph_dummy = comm_dummy.environment_graph()
            success1_dummy, message_dummy, success2_dummy, graph_dummy = clean_graph(comm_dummy, graph_dummy, ['chicken'])

            # Add human
            comm_dummy.add_character('Chars/Female1', initial_room='kitchen')
            # Add ad hoc agent
            comm_dummy.add_character('Chars/Male1', initial_room='kitchen')
            # Add ad hoc agent
            comm_dummy.add_character('Chars/Female2', initial_room='kitchen')

            for script_instruction in current_script:
                act_success, success_msgs, message = comm_dummy.render_script([script_instruction], recording=False, skip_animation=True)
            
            # only the last instruction is related
            if '|' in script_instruction:
                script_split = script_instruction.split('|')
                action_map = {f'<char{i}>': f'act{i+1}' for i in range(num_agents)}  # 4 characters
                action_vars = {f'act{i+1}': None for i in range(num_agents)}
                for part in script_split:
                    for char, action in action_map.items():
                        if char in part:
                            action_vars[action] = part  # Assign the string part to the corresponding action variable
                            break
                act1, act2, act3 = action_vars.values()
            else:
                # single action (no '|')
                action_map = {f'<char{i}>': f'act{i+1}' for i in range(num_agents)}  # 4 characters
                action_vars = {f'act{i+1}': None for i in range(num_agents)}
                for char, action in action_map.items():
                    if char in script_instruction:
                        action_vars[action] = script_instruction
                        break
                act1, act2, act3 = action_vars.values()
            acts = [act1, act2, act3]

            # change to a loop later
            for agent_idx in range(num_agents):
                if acts[agent_idx] and success_msgs[agent_idx]:
                    all_prev_actions[agent_idx].pop(0)
                    all_prev_actions[agent_idx].append(acts[agent_idx])
                else:
                    all_prev_actions[agent_idx].pop(0)
                    all_prev_actions[agent_idx].append('None')
            # else:
            #     del current_script[-1]
            success, graph = comm_dummy.environment_graph()
    return future_actions, graphs, actual_predict_time

# convert action predictions for the other agents to fluent literals of ASP
def get_fluents(ASP_goal, other_actions, graphs, fluents, id_dict, num_agents, current_agent_id):
    future_actions = []
    exo_dict = {
        'grab': 'exo_grab',
        'putback': 'exo_put',
        'putin': 'exo_put',
        'open': 'exo_open',
        'close': 'exo_close',
        'switchon': 'exo_switchon',
        'switchoff': 'exo_switchoff'
    }
    aspname_dict = {
        'cereal': 'cereal',
        'breadslice': 'breadslice',
        'bananas':'bananas',
        'plum':'plum',
        'cupcake': 'cupcake',
        'cutlets': 'cutlets',
        'chips': 'chips',
        'plate': 'plate',
        'candybar': 'candybar',
        'milk': 'milk',
        'wine': 'wine',
        'juice': 'juice',
        'computer': 'computer',
        'coffeemaker': 'coffeemaker',
        'waterglass': 'waterglass',
        'mug': 'mug',
        'coffeepot': 'coffeepot',
        'kitchentable': 'kitchentable',
        'book': 'book',
        'boardgame': 'boardgame',
        'stove': 'stove',
        'cellphone': 'cellphone',
        'apple': 'apple',
        'bookshelf': 'livingroom_desk',
        'microwave': 'kitchen_smalltable',
        'chair':'livingroom_desk',
        'desk':'bedroom_desk',
        'tvstand':'kitchen_smalltable',
        'kitchencounter':'counter_one',
        'dishwasher':'counter_three',
        'floor':'livingroom_desk',
        'sofa':'livingroom_coffeetable'
    }
    duplicate_aspname_dict = {
        'coffeetable': ['livingroom_coffeetable', 'bedroom_coffeetable']
    }

    goal_spl = ASP_goal.split(',I),')
    goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
    for agent_id in range(num_agents):
        if agent_id != current_agent_id:
            object_ids = [edge['to_id'] for edge in graphs[0]['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    if object_ids:
        for human_obj in object_ids:
            name_ = list(id_dict.keys())[list(id_dict.values()).index(human_obj)] # performance >_<
            goal_spl = [item for item in goal_spl if name_ not in item]
        ASP_goal = ','.join(goal_spl)
    # print(ASP_goal)
    if not other_actions:
        return ASP_goal, fluents

    for idx, action_list in enumerate(other_actions): # other_act = [[None,None,None],[None,None,None]]
        for agent in range(num_agents-1):
            agent_id = agent if agent < current_agent_id else agent+1
            other_act = action_list[agent]
            if other_act:
                # convert to exo
                action_spl = other_act.split(' ')
                act_type = action_spl[1][1:-1]
                act_obj = action_spl[2][1:-1]
                if len(action_spl) == 4 and act_type != 'find': # find has low probability
                    if act_obj in aspname_dict:
                        obj = aspname_dict[act_obj]
                    else:
                        objs = duplicate_aspname_dict[act_obj]
                        obj_proximity = []
                        other_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['class_name'] == 'character'][agent_id]
                        for obj in objs:
                            id_ = id_dict[obj]
                            obj_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['id'] == id_][0]
                            prox_human = np.linalg.norm(np.asarray(other_pose['position'])-np.asarray(obj_pose['position']))
                            obj_proximity.append(prox_human)
                        proxmin_idx = obj_proximity.index(min(obj_proximity)) # take the less, if equal the first one
                        obj = objs[proxmin_idx]           
                    
                    future_action = 'occurs(' + exo_dict[act_type] + '('+ ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) +',' + obj + '),' + str(idx) + ').'
                    future_actions.append(future_action)
                    if act_type == 'grab' and obj in ASP_goal: # ignore move/open/close/switchon/switchoff
                        goal_spl = ASP_goal.split(',I),')
                        goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
                        goal_spl = [item for item in goal_spl if obj not in item]
                        ASP_goal = ','.join(goal_spl)
                elif len(action_spl) > 4: # put
                    act_type = action_spl[1][1:-1]
                    act_obj = action_spl[2][1:-1]
                    act_loc = action_spl[4][1:-1]
                    if act_obj in aspname_dict:
                        obj = aspname_dict[act_obj]
                    else:
                        objs = duplicate_aspname_dict[act_obj]
                        # one or two
                        keys = []
                        if object_ids:
                            for human_obj in object_ids:
                                key_ = list(id_dict.keys())[list(id_dict.values()).index(human_obj)] # performance >_<
                                if key_ in objs:
                                    keys.append(key_)
                            obj = keys[0] if keys else None
                        else:
                            obj = None
                    if act_loc in aspname_dict:
                        loc = aspname_dict[act_loc]
                    else:
                        locs = duplicate_aspname_dict[act_loc]
                        loc_proximity = []
                        human_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['class_name'] == 'character'][agent_id]
                        for loc in locs:
                            id_ = id_dict[loc]
                            loc_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['id'] == id_][0]
                            prox_human = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(loc_pose['position']))
                            loc_proximity.append(prox_human)
                        proxmin_idx = loc_proximity.index(min(loc_proximity)) # take the less, if equal the first one
                        loc = locs[proxmin_idx]  
                    if obj:
                        future_action = 'occurs(' + exo_dict[act_type] + '(' +('human' if agent_id == 0 else 'ahagent'+str(agent_id))+ ',' + obj + ',' + loc + '),' + str(idx) + ').'
                        future_actions.append(future_action)
    # print(future_actions)
    return ASP_goal, fluents+future_actions

# map LLM text to goal text
def map_LLM_text_ASP_goal(LLM_goal):
    llm_goal_dict = {
        'Prepare breakfast weekday':'Breakfast_weekday',
        'Prepare coffee':'Coffee',
        'Prepare home work-station':'Workstation',
        'Prepare lunch':'Lunch',
        'Pack bag':'Pack_bag',
        'Clean kitchen':'Clean_kitchen',
        'Prepare breakfast weekend':'Breakfast_weekend',
        'Prepare table for guests':'Make_table',
        'Serve snacks':'Serve_snacks',
        'Prepare activities':'Activities',
        'Clean dishes':'Clean_dishes'
    }
    return llm_goal_dict[LLM_goal]

# map goal to ASP goal
def map_goal_ASP(goal):
    goal_dict = {
        'Breakfast_weekday': 'holds(on(cereal,kitchentable),I), holds(on(breadslice,kitchentable),I), holds(on(bananas,kitchentable),I)',
        'Coffee': 'holds(made(coffee),I), holds(on(coffeepot,livingroom_desk),I), holds(on(mug,livingroom_desk),I)',
        'Workstation': 'holds(switchedon(computer),I), holds(on(book,livingroom_desk),I), holds(on(cellphone,livingroom_desk),I)',
        'Lunch': 'holds(on(waterglass,kitchentable),I), holds(on(apple,kitchentable),I), holds(on(chips,kitchentable),I)',
        'Pack_bag': 'holds(on(juice,livingroom_desk),I), holds(on(cellphone,livingroom_desk),I)',
        'Clean_kitchen': 'holds(on(cereal,livingroom_desk),I), holds(on(milk,kitchen_smalltable),I), holds(on(cutlets,bedroom_desk),I)',
        'Breakfast_weekend': 'holds(on(cereal,kitchentable),I), holds(on(breadslice,kitchentable),I), holds(on(cupcake,kitchentable),I)',
        'Make_table': 'holds(on(plate,kitchentable),I), holds(on(wine,kitchentable),I)',
        'Serve_snacks': 'holds(on(plum,kitchentable),I), holds(on(candybar,kitchentable),I)',
        'Activities': 'holds(on(book,livingroom_coffeetable),I), holds(on(boardgame,livingroom_coffeetable),I)',
        'Clean_dishes': 'holds(on(plate,dishwasher),I), holds(on(mug,dishwasher),I), holds(switchedon(dishwasher),I)'
    }
    return goal_dict[goal]

# get goal objects
def get_goal_obj(goal):
    goal_dict = {
        'Breakfast_weekday': ['cereal', 'breadslice', 'bananas'],
        'Coffee': ['coffeepot', 'mug'],
        'Workstation': ['computer', 'book', 'cellphone'],
        'Lunch': ['waterglass', 'apple', 'chips'],
        'Pack_bag': ['juice', 'cellphone'],
        'Clean_kitchen': ['cereal', 'milk', 'cutlets'],
        'Breakfast_weekend': ['cereal', 'breadslice, cupcake'],
        'Make_table': ['plate, wine'],
        'Serve_snacks': ['plum', 'candybar'],
        'Activities': ['book', 'boardgame'],
        'Clean_dishes': ['plate', 'mug', 'dishwasher']
    }
    if isinstance(goal,list):
        objs = []
        for task in goal:
            goal = map_LLM_text_ASP_goal(task)
            objs = objs + goal_dict[goal]
        return objs
    return goal_dict[goal]

def remove_excess_fluents(ASP_goal):
    goal_spl = ASP_goal.split(',I),')
    goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
    if len(goal_spl) > 3:
        goal_spl = goal_spl[:4]
        ASP_goal = ','.join(goal_spl)
    return ASP_goal

# get the human interpretation of the other agent acitons
def get_human_interpretation(graph, action):
    action_dict = {
        'move': '{} moved to {}',
        'grab': '{} grabbed {}',
        'put': '{} put {} in {}',
        'open': '{} opened {}',
        'close': '{} closed {}',
        'switchon': '{} switched on {}',
        'switchoff': '{} switched off {}'
    }

    if action:
        action_match = re.match(r"occurs\((\w+)\(([^)]+)\),\d+\)", action)
        if not action_match:
            print('Invalid format')
            return None, None
        
        verb, objects = action_match.groups()
        objects = objects.split(',')

        if verb in action_dict:
            # get action cause
            exec = get_precondition(graph, verb, objects)
            temp = action_dict[verb]
            return temp.format(*objects), exec
        else:
            print('Action is of unknown format.')
            return None, None
    return action, None

# get the human interpretation of the other agent action and causal effect
def get_causal_action_literals(llm_out):
    action_pattern = r'([a-zA-Z_]+\([\w\s,]+\))\s+causes'
    literal_pattern = r'causes\s+([a-zA-Z_]+\([\w\s,]+\))'

    action_match = re.search(action_pattern, llm_out)
    literal_match = re.search(literal_pattern, llm_out)

    action = action_match.group(1).lower() if action_match else None
    literals = literal_match.group(1).lower() if literal_match else None
    return action, literals

# get the human interpretation of the other agent action and causal effect
def get_exec_action_literals(exec_axiom):
    action, body_lit = exec_axiom.split('if',1)
    literals = re.findall(r'\b\w+\([^)]+\)', body_lit)
    literals = [literal.strip() for literal in literals]
    return action[1:].strip(), literals

# get the human interpretation of the other agent actions and objects
def get_action_objects(action):
    action_name, objects = action.split("(", 1)
    objects = objects.rstrip(")").split(",")
    objects = [obj.strip() for obj in objects]
    return action_name, objects

# identify higher level sort the new obj belongs to
def calssify_with_llm(sort_list, new_sort_names):
    prompt = '''The agent's knowledge base consists of the following sorts: 
    asp_sorts list ''' + sort_list + ''' . The new objects ''' + ', '.join(new_sort_names) + ''' needs 
    to be classified into one or more of these sorts. Based on common usage, which 
    sorts do they most likely belong to? 
    Provide your answer in the following format with no extra explanation:
    {'new object': ['sort1', 'sort2']}'''        
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# find the lowest category
def get_lowerst_category(categories, sort_list):
    lists_dict, combined_lists = {}, {}
    for sort_line in sort_list:
        key, value = sort_line.split('=',1)
        key = key.strip()
        value = value.strip()
        if '+' in value: # high level sort list
            combined_lists[key] = [val_list.strip() for val_list in value.split('+')]
        else:
            lists_dict[key] = ast.literal_eval(value) # for normal lists
    
    lowest_cat = next((category for line in sort_list for category in categories if category in line), categories[0])
    for sub_category in categories:
        sub_category = sub_category.replace('#','')
        if sub_category in lists_dict and sub_category not in combined_lists:
            lowest_cat = sub_category
    return lowest_cat

# write the new object to base file
def update_sort_file(categories, sort_list, file_path):
    updated_lines, write_back = {}, []
    
    for key, value in categories.items():
        if len(value) > 1:
            categories[key] = [get_lowerst_category(value, sort_list)]

    for sort_line in sort_list:
        key, value = sort_line.split('=',1)
        key = key.strip()
        value = value.strip()
        for key_, value_ in categories.items(): # key_ = new_object
            if key in updated_lines and key == value_[0]: # need to chec this next
                updated_value = updated_lines[key]
                if '+' in updated_value:
                    current_list = None
                    for val_list in updated_value.split('+'):
                        val_list = val_list.strip()
                        try:
                            val_list = ast.literal_eval(val_list)
                            # if isinstance(val_list, list):
                            val_list.append(key_)
                            val_list = str(val_list)
                        except (ValueError, SyntaxError):
                            pass
                        if current_list:
                            current_list += ' + ' + val_list
                        else:
                            current_list = val_list
                else:
                    # current_list = ast.literal_eval(updated_value)
                    current_list = updated_value
                    current_list.append(key_)
                updated_lines[key] = current_list
            elif key == value_[0]:
                if '+' in value:
                    current_list = None
                    for val_list in value.split('+'):
                        val_list = val_list.strip()
                        try:
                            val_list = ast.literal_eval(val_list)
                            # if isinstance(val_list, list):
                            val_list.append(key_)
                            val_list = str(val_list)
                        except (ValueError, SyntaxError):
                            pass
                        if current_list:
                            current_list += ' + ' + val_list
                        else:
                            current_list = val_list
                else:
                    current_list = ast.literal_eval(value)
                    current_list.append(key_)
                updated_lines[key] = current_list
    
    for sort_line in sort_list:
        key, value = sort_line.split('=',1)
        key = key.strip()
        value = value.strip()
        if key in updated_lines:
            write_back.append(f"{key} = {updated_lines[key]}\n")
        else:
            write_back.append(f"{key} = {value}\n")

    with open(file_path, 'w') as file:
        file.writelines(write_back)

    return categories

# get the sort category an object belongs to
def get_sort_categories(new_objects):
    categories = {}
    sort_line = get_sortlist()
    for obj in new_objects:
        categories[obj] = []
        for key, values in sort_line.items():
            if obj in values:
                categories[obj].append(key)
    # set agent objects category as #agent since they need to be replaced by the actual agent
    for key, value in categories.items():
        if 'agent' in key and not value:
            categories[key].append('agent')
    return categories

# return all actions in the ASP file
def get_all_actions():
    with open('ASP/ahagent_pre2.sp', 'r') as file:
        all_actions = []
        capturing =  False
        for line in file:
            if line.startswith('#agent_actions'):
                capturing = True
                key, actions = line.split('=', 1)
                all_actions.extend(act.strip() for act in actions.split('+'))
            elif capturing and (line.startswith('#') or line.startswith('predicates')):
                break
            elif capturing:
                all_actions.extend(act.strip() for act in line.split('+')) 

    all_actions = [act.rstrip('.') for act in all_actions if act.strip()]
    return all_actions

# check if an action already exisits in KB
def check_action_exisits(action):
    action_exists = False
    action_select = None
    all_actions = get_all_actions()
    for act in all_actions:
        if act.startswith(action):
            action_exists = True
            action_select = act
        # elif '_' in action: # wordnet replace part
        #     revised_verb = action.split('_')[0]
        #     if act.startswith(revised_verb):
        #         action_exists = True
        #         action_select = act
    return action_exists, action_select

# update the asp file with new actions or updated versions of the actions
def update_action_variations(llm_action, action_verb, categories_dict):
    all_combinations, all_actions = [], []
    new_action = None
    # secnario 1 -> acton exisits with same sorts(or high sorts)
    # scenario 2 -> action exisits with low sorts
    # scenario 3 -> action does not exist

    action_exists, action_select = check_action_exisits(action_verb)
    
    all_combinations.extend(list(product(*categories_dict.values())))
    objects = (llm_action.split("(", 1)[1]).rstrip(")").split(",")
    for combo in all_combinations:
        temp_action = llm_action
        for key, comb in zip(objects, combo):
            temp_action = temp_action.replace(key, '#'+comb)
        all_actions.append(temp_action)
    
    if all_actions:
        if action_exists:
            if action_select in all_actions: # action exisit, and with same sort group
                print('Action already exists with the same sort group')
                new_action = action_select
            else: # action exsits, but with a lower sort groups, need to raise the sort level
                old_sorts = (action_select.split("(", 1)[1]).rstrip(")").split(",")
                temp_action = all_actions[0]
                new_sorts = (temp_action.split("(", 1)[1]).rstrip(")").split(",")
                all_sorts = get_sortlist()
                for old_, new_ in zip(old_sorts, new_sorts):
                    if old_ !=  new_:
                        old_key = old_.strip('#')
                        old_sort_values = all_sorts[old_key]
                        new_key = new_.strip('#')
                        new_sort_values = all_sorts[new_key]
                        old_new_sort_values = old_sort_values + new_sort_values

                        for key, value in all_sorts.items():
                            # lets find the first sort list which has the combination of sorts
                            if set(old_new_sort_values).issubset(value):
                                temp_action = temp_action.replace(new_,'#'+key)
                                break
                new_action = temp_action
                with open('ASP/ahagent_pre2.sp','r') as file:
                    file_lines = file.readlines()
                updated_lines = []
                capturing = False
                for line in file_lines:
                    if line.startswith('#agent_actions'):
                        capturing = True
                        line = line.replace(action_select, temp_action)
                    else:
                        if capturing and (line.startswith('#') or line.startswith('predicates')):
                            capturing = False
                        elif capturing:
                            line = line.replace(action_select, temp_action)
                    updated_lines.append(line)
                with open('ASP/ahagent_pre2.sp','w') as file:
                    file.writelines(updated_lines)
                # write action to action file to be explored later
                with open('tobe_explored_actions.txt', 'a') as file:
                    file.write(all_actions[0])
        else:
            # action does not exist, learn the action with the lowest sort level you find for that particular object
            new_action = all_actions[0]
            with open('ASP/ahagent_pre2.sp','r') as file:
                file_lines = file.readlines()
            updated_lines = []
            capturing = False
            for line in file_lines:
                if line.startswith('#agent_actions'):
                    capturing = True
                else:
                    if capturing and (line.startswith('#') or line.startswith('predicates')):
                        capturing = False
                    elif capturing:
                        line = line.rstrip()[:-1] + ' + ' + new_action + '. \n'
                updated_lines.append(line)
            with open('ASP/ahagent_pre2.sp','w') as file:
                file.writelines(updated_lines)
            with open('tobe_explored_actions.txt', 'a') as file:
                file.write(new_action)
    return new_action

# return all fluents in the ASP file
def get_all_fluents():
    with open('ASP/ahagent_pre2.sp', 'r') as file:
        all_fluents = []
        capturing =  False
        for line in file:
            if line.startswith('#inertial_f') or line.startswith('#defined_f'):
                capturing = True
                key, literals = line.split('=', 1)
                all_fluents.extend(lit.strip() for lit in literals.split('+'))
            elif capturing and (line.startswith('#') or line.startswith('predicates')):
                break

    all_fluents = [lit.rstrip('.') for lit in all_fluents if lit.strip()]
    return all_fluents

def split_expression(expression):
    head, body = expression[:-1].split(":-", 1)
    body_list = re.split(r',(?![^\(\)]*\))', body.strip())
    return head.strip(), [part.strip() for part in body_list]

def check_nested_parentheses(literal):
    open_paren_count = literal.count('(')
    close_paren_count = literal.count(')')
    return open_paren_count > 1 or close_paren_count > 1

def add_exec_condition(exec_action, exec_axiom, exec_literals):
    sort_dict = get_sortlist()
    sort_dict['agent'] = ['ahagent1', 'ahagent2']
    valid_axiom = False
    
    # process the literals from exec_axiom
    head_lit, body_lit = exec_axiom.split('if',1)
    body_lits = re.findall(r'\b\w+\([^)]+\)', body_lit)
    exec_axiom = exec_axiom.replace(head_lit, '-occurs('+ head_lit[1:].replace(' ','') + ',I) ')
    for lit in body_lits:
        if 'next_to' in lit:
            exec_axiom = exec_axiom.replace(lit, lit.replace(' ',''))
        else:
            exec_axiom = exec_axiom.replace(lit, 'holds('+ lit.replace(' ','') + ',I)')

    head_lit_dict, body_lits_dict, categories, head_comb_dict, body_comb_dict, obj_categories = {}, {}, {}, {}, {}, {}
    
    head_verb, head_objects = get_action_objects(head_lit[1:].strip())
    head_lit_dict[head_lit[1:].strip()] = [obj.strip() for obj in head_objects] # size 1 dict since always one action

    for body_lit in body_lits:
        if not check_nested_parentheses(body_lit):
            body_objects = (body_lit.strip().split("(", 1)[1]).rstrip(")").split(",")
            body_lits_dict[body_lit] = [obj.strip() for obj in body_objects]
    
    if head_lit_dict and body_lits_dict:
        # replace the grounded terms in the literals with their sort values
        for key, value in head_lit_dict.items():
            for obj in value:
                categories[obj] = []
                for key, values in sort_dict.items():
                    if obj in values and key not in categories[obj]:
                        categories[obj].append('#'+key)
        
        for key, value in body_lits_dict.items():
            for obj in value:
                if not (obj in categories):
                    categories[obj] = []
                for key, values in sort_dict.items():
                    if obj in values and '#'+key not in categories[obj]:
                        categories[obj].append('#'+key)

    for key, value in head_lit_dict.items(): # -move(ahagent1, kitchentable)': ['ahagent1', 'kitchentable']
        types_list = [categories.get(val, [val]) for val in value]
        all_combinations = list(product(*types_list))
        pred_name = key.split('(')[0]
        new_literals = [
            f"{pred_name}({','.join(comb)})"
            for comb in all_combinations
        ]
        head_comb_dict[key] = new_literals

    for key, value in body_lits_dict.items(): # at(ahagent1, kitchentable): ['ahagent1', 'kitchentable']
        types_list = [categories.get(val, [val]) for val in value]
        all_combinations = list(product(*types_list))
        pred_name = key.split('(')[0]
        new_literals = [
            f"{pred_name}({','.join(comb)})"
            for comb in all_combinations
        ]
        body_comb_dict[key] = new_literals

    # check whether each action and fluient exits in the knowledge base
    all_fluents = get_all_fluents()
    all_fluents.append('next_to(#furniture_surfaces,#furniture_surfaces)') # predicate

    selected_body_lits = []
    for key_, value_ in body_comb_dict.items():
        for lit_combo in value_:
            if lit_combo in all_fluents and lit_combo not in selected_body_lits: # fluents alerady exisits in asp
                selected_body_lits.append(lit_combo)                            

    if len(body_lits) == len(selected_body_lits):
        # all head and body literals are in the knowledge base - axiom valid and can be added
        valid_axiom = True
        act_combo_objects = (exec_action.strip().split("(", 1)[1]).rstrip(")").split(",")
        head_obj_idx = 0       
        for obj in head_objects:
            if obj not in obj_categories:
                obj_categories[obj.strip()] = act_combo_objects[head_obj_idx]
            else:
                obj_categories[obj].extend(act_combo_objects[head_obj_idx])
            head_obj_idx += 1

        body_lit_idx = 0
        for body_lit in body_lits:
            body_lit_objects = (selected_body_lits[body_lit_idx].strip().split("(", 1)[1]).rstrip(")").split(",")
            body_obj_idx = 0
            for obj in body_lits_dict[body_lit]:
                if obj not in obj_categories:
                    obj_categories[obj.strip()] = body_lit_objects[body_obj_idx]
                else:
                    file_path = 'asp_sorts_partial.txt'
                    with open(file_path, 'r') as file:
                        sort_list = file.read()
                    obj_categories[obj] = get_lowerst_category([obj_categories[obj], body_lit_objects[body_obj_idx]], sort_list.splitlines())
                body_obj_idx += 1
            body_lit_idx += 1

    sorts_dict = {
        'R': 'agent', 'F': 'food', 'D': 'drinks',
        'E': 'electricals', 'A': 'appliances',
        'P': 'plates', 'N': 'glasses', 'C': 'containers',
        'T': 'others', 'L': 'furniture_surfaces', 'S': 'surfaces', 
        'G':  'graspable', 'O': 'objects'
    }
    if valid_axiom:
        for key, value in obj_categories.items():
            exec_axiom = exec_axiom.replace(key, value)
        for key, value in sorts_dict.items():
            exec_axiom = exec_axiom.replace(value, key)
        exec_axiom = exec_axiom.replace('#', '').replace('if', ':-') + '.'
        # check special
        if 'next_to' in exec_axiom:
            exec_axiom = exec_axiom.replace('next_to(L,L)', 'next_to(L1,L2)')
            head, body = exec_axiom.split(':-')
            head = head.replace('L)', 'L1)')
            body = body.replace('L)', 'L2)')
            exec_axiom = head + ':-' + body
        new_head, new_body = split_expression(exec_axiom)
        # new_body = [par for itm in new_body for par in itm.split(' ')]
        new_body = [itm for itm in new_body if (itm.startswith('hold') or itm.startswith('next') or itm.startswith('not'))]
        
        with open('ASP/ahagent_pre2.sp','r') as file:
            file_lines = file.readlines()

        start_tag = '-------------------- executability conditions -------------------'
        end_tag = '------------------------ inertial axioms ------------------------'

        capturing = False
        updated_lines = []
        axiom_exist = False
        for line in file_lines:
            if start_tag in line:
                capturing = True
                updated_lines.append(line)
                continue
            elif end_tag in line:
                # if axiom does not exist add the new axiom here
                if not axiom_exist:
                    updated_lines.append('% added during run from human cue.\n')
                    updated_lines.append(exec_axiom+'\n')
                    print(exec_axiom+'\n')
                capturing = False
                updated_lines.append(line)
                continue

            if capturing:
                if line.strip() == '' or line.startswith('%'):
                    updated_lines.append(line)
                else:
                    # check if axiom exist
                    line_head, line_body = split_expression(line)
                    if line_head == new_head and set(new_body).issubset(line_body):                        
                        axiom_exist = True
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        with open('ASP/ahagent_pre2.sp','w') as file:
            file.writelines(updated_lines)

    return

# add causal laws for new actions
def add_causal_axiom(action, llm_action, cause_literal):
    sorts_dict = {
        'R': 'agent', 'F': 'food', 'D': 'drinks',
        'E': 'electricals', 'A': 'appliances',
        'P': 'plates', 'N': 'glasses', 'C': 'containers',
        'T': 'others', 'L': 'furniture_surfaces', 'S': 'surfaces', 
        'G':  'graspable', 'O': 'objects'
    }

    llm_terms = re.findall(r'\w+', llm_action)
    action_terms = re.findall(r'\w+', action)
    for key, value in zip(llm_terms, action_terms):
        try:
            cause_literal = cause_literal.replace(key, value)
        except AttributeError as e:
            return

    # this need to be replaced with the wordnet later so we can learn axioms even with unknown(new) fluents
    all_fluents = get_all_fluents()

    match = re.match(r'(\w+)\(', cause_literal)
    valid_axiom = False

    if match:
        fluent_name = match.group(1)
        for fluent in all_fluents:
            flu_name = fluent.split('(')[0]
            if fluent_name == flu_name:
                new_fluent_sorts = (cause_literal.strip().split("(", 1)[1]).rstrip(")").split(",")
                new_fluent_sorts = [sor.strip() for sor in new_fluent_sorts]
                fluent = fluent.replace('#', '')
                old_fluent_sorts = (fluent.strip().split("(", 1)[1]).rstrip(")").split(",")
                old_fluent_sorts = [sor.strip() for sor in old_fluent_sorts]
                file_path = 'asp_sorts_partial.txt'
                with open(file_path, 'r') as file:
                    sort_list = file.read()

                lists_dict, combined_lists = {}, {}
                for sort_line in sort_list.splitlines():
                    key, value = sort_line.split('=',1)
                    key = key.strip()
                    value = value.strip()
                    if '+' in value: # high level sort list
                        combined_lists[key] = [val_list.strip() for val_list in value.split('+')]
                    else:
                        lists_dict[key] = ast.literal_eval(value) # for normal lists
                
                for obj_sort in new_fluent_sorts:
                    for key, value in combined_lists.items():
                        if obj_sort in value and key in old_fluent_sorts:
                            fluent = fluent.replace(key,obj_sort)
                cause_literal = fluent
                valid_axiom = True

    if valid_axiom:
        for key, value in sorts_dict.items():
            action = action.replace(value, key)
            cause_literal = cause_literal.replace(value, key)

        causal_axiom = 'holds(' +cause_literal+ ',I+1) :- occurs(' +action.replace('#','')+ ',I).\n'
        new_head, new_body = split_expression(causal_axiom)

        with open('ASP/ahagent_pre2.sp','r') as file:
            file_lines = file.readlines()

        capturing = False
        updated_lines = []
        axiom_exist = False

        start_tag =  '------- casual laws -----'
        end_tag = 'exogeneous actions effect'

        for line in file_lines:
            if start_tag in line:
                capturing = True
                updated_lines.append(line)
                continue
            elif end_tag in line:
                # if axiom does not exist add the new axiom here
                if not axiom_exist:
                    updated_lines.append('% added during run from human cue.\n')
                    updated_lines.append(causal_axiom+'\n')
                    print(causal_axiom+'\n')
                capturing = False
                updated_lines.append(line)
                continue

            if capturing:
                if line.strip() == '' or line.startswith('%'):
                    updated_lines.append(line)
                else:
                    # check if axiom exist
                    line_head, line_body = split_expression(line)
                    if line_head == new_head and set(new_body[:-1]).issubset(line_body):
                        axiom_exist = True
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
            
        with open('ASP/ahagent_pre2.sp','w') as file:
            file.writelines(updated_lines)
    return

def learn_action(act):
    file_path = 'asp_sorts_partial.txt'
    # seperate the action verb and objects
    action_verb, all_objects = get_action_objects(act)
    print(action_verb, all_objects)

    # examine whether the objects are known
    new_sort_names = []
    sort_dict = get_sortlist()
    sort_surfaces = sort_dict['surfaces']
    sort_objects = sort_dict['objects']
    asp_sort_list = sort_surfaces + sort_objects

    for obj in all_objects:
        obj = obj.strip()
        if obj not in asp_sort_list and 'agent' not in obj:
            new_sort_names.append(obj)

    # if any new object(sort) names, then add to the current knowledge base
    if new_sort_names:
        # find the higher level sort the new object belong to
        with open(file_path, 'r') as file:
            sort_list = file.read()

        category = calssify_with_llm(sort_list, new_sort_names)
        response_dict = eval(category)
        # add the new object to sort file
        _ = update_sort_file(response_dict, sort_list.splitlines(), file_path)
    
    final_categories_dict = get_sort_categories(all_objects)
    new_action = update_action_variations(act, action_verb, final_categories_dict)
    return new_action

# retrive actions and axioms from cues and update the knowledge base
# assumption: each cue only descibe one action at a time
def update_action_axioms(cue, exec):
    # convert the human cue to action causal effect format using llms
    exec_axiom, causal_axiom = convert_cue_to_axiom(cue, exec)
    print(exec_axiom, causal_axiom)
    
    # seperate the action and causal literals
    causal_action, cause_literals = get_causal_action_literals(causal_axiom)
    
    # seperate the action and preconditions
    exec_action, exec_literals = get_exec_action_literals(exec_axiom) # exec_literals is a list

    new_causal_action = learn_action(causal_action)
    new_exec_action = learn_action(exec_action)

    if new_causal_action: # does not mean the new action was discovered in this trial - action exisit in ASP
        # new action found, learn the causal law for that action
        add_causal_axiom(new_causal_action, causal_action, cause_literals)
    if new_exec_action:
        add_exec_condition(new_exec_action, exec_axiom, exec_literals)
    return

# get taskpredictions from LLM
def convert_cue_to_axiom(cue, exec):
    system_msg = '''You are a formal logic translator. Your task is to convert natural language explanations into ASP-style formal rules.
    Use the following syntax:

    Causal Law:
    action(agent, object) causes fluent.
    or
    action(agent, object, target) causes fluent.

    Executability Condition (i.e., when the action is not possible):
    -action(agent, object) if condition1, condition2.

    Only use the predicates present in the input. Stick to known formats.
    Do not create new variables, assumptions, or modify the sentence meaning.
    '''
    
    user_msg1 = 'ahagent1 cannot put the cake inside the microwave since the door is closed. Thus ahagent1 opened the microwave.'
    assistant_msg1 = 'exec_cond: -put(ahagent1, cake, microwave) if not opened(microwave). causal: open(ahagent1, microwave) causes opened(microwave).'

    user_msg2 = 'ahagent1 cannot grab the waterbottle since the waterbottle is on the kitchentable and the ahagent1 is not at kitchentable. Thus ahagent1 moved to kitchentable.'
    assistant_msg2 = 'exec_cond: -grab(ahagent1, waterbottle) if on(waterbottle, kitchentable), not at(ahagent1, kitchentable).  causal: move(ahagent1, kitchentable) causes at(ahagent1, kitchentable).'
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg1},
            {"role": "assistant", "content": assistant_msg1},
            {"role": "user", "content": user_msg2},
            {"role": "assistant", "content": assistant_msg2},
            {"role": "user", "content": 'Now convert the following: ' + exec + 'Thus ' + cue}
        ]
    )

    print(exec + 'Thus ' + cue)

    print(completion.choices[0].message.content)
    # pattern = r"\[.*?\]"
    # match_ = re.search(pattern,completion.choices[0].message.content)
    llm_output =  completion.choices[0].message.content # ast.literal_eval(match_.group(0))
    exec_axiom_match = re.search(r'exec_cond:\s*(.*?)\s*\.', llm_output)
    causal_axiom_match = re.search(r'causal:\s*(.*?)\s*\.', llm_output)
    exec_axiom = exec_axiom_match.group(1).strip() if exec_axiom_match else None
    causal_axiom = causal_axiom_match.group(1).strip() if causal_axiom_match else None
    return clean_action_names(exec_axiom), clean_action_names(causal_axiom)

def clean_action_names(line):
    if line:
        cleaned_line = re.sub(r'(?<![\w\(])(-?\w+)_+(\w+)(?=\()', lambda m: m.group(1).replace('-', '') + m.group(2) if m.group(1).startswith('-') else m.group(1) + m.group(2), line)
        if line.strip().startswith('-') and not cleaned_line.strip().startswith('-'):
            cleaned_line = '-' + cleaned_line
    else:
        cleaned_line = line
    return cleaned_line

def list_to_sorts(item_list, name):
    items_str = ', '.join(item_list)
    sorts = f'#{name} = {{{items_str}}}'
    return sorts

def join_sorts(variables):
    valid_variable_names = [name for name, item in variables if item]
    temp_graspable = ''
    if valid_variable_names:
        temp_graspable = ' + '.join(valid_variable_names)
    return temp_graspable

def get_sorts_objects(ASP_goal, fluents):
    temp_objects = []
    sorts = ['sorts']
    items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',ASP_goal)] + ['plate', 'waterglass', 'coffeepot']
    for fluent in fluents:
        if fluent.startswith('occurs') or 'grabbed' in fluent or 'agent_hand' in fluent: # exo actions, other aegnts hand or agent hands
            fitems = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            fitems = [item.split(',')[1] if ',' in item else item for item in fitems]
            if fitems[0] not in items:
                items.append(fitems[0])
    fir_items = [item.split(',')[0] if ',' in item else item for item in items]
    sec_items = [item.split(',')[1] if ',' in item else item for item in items]
    sort_dict = get_sortlist()
    locations = sort_dict['furniture_surfaces']
    items = fir_items + [item for item in sec_items if item not in locations and item not in fir_items]
    # food
    food = sort_dict['food']
    temp_food = [item for item in food if item in items]
    if not temp_food:
        temp_food = [food[0]]
    temp_objects += temp_food
    if temp_food:
        temp_food = list_to_sorts(temp_food, 'food')
        sorts.append(temp_food+'.')
    # drinks
    drinks = sort_dict['drinks']
    temp_drinks = [item for item in drinks if item in items] + ['coffee']
    temp_objects += temp_drinks
    if temp_drinks:
        temp_drinks = list_to_sorts(temp_drinks, 'drinks')
        sorts.append(temp_drinks+'.')
    # electricals
    electricals = sort_dict['electricals']
    temp_electricals = [item for item in electricals if item in items] + ['dishwasher']
    temp_objects += temp_electricals
    temp_electricals = list_to_sorts(temp_electricals, 'electricals')
    sorts.append(temp_electricals+'.')
    # appliances
    appliances = sort_dict['appliances']
    temp_appliances = [item for item in appliances if item in items and item not in electricals] + ['coffeemaker']
    temp_objects += temp_appliances
    temp_appliances = list_to_sorts(temp_appliances, 'appliances') + ' + #electricals'
    sorts.append(temp_appliances+'.')
    # plates
    plates = sort_dict['plates']
    temp_plates = [item for item in plates if item in items]
    if not temp_plates:
        temp_plates = [plates[0]]
    temp_objects += temp_plates
    if temp_plates:
        temp_plates = list_to_sorts(temp_plates, 'plates')
        sorts.append(temp_plates+'.')
    # glasses
    glasses = sort_dict['glasses']
    temp_glasses = [item for item in glasses if item in items]
    if not temp_glasses:
        temp_glasses = [glasses[0]]
    temp_objects += temp_glasses
    if temp_glasses:
        temp_glasses = list_to_sorts(temp_glasses, 'glasses')
        sorts.append(temp_glasses+'.')
    # containers
    containers = sort_dict['containers']
    temp_containers = [item for item in containers if item in items and item not in plates and item not in glasses] + ['coffeepot']
    temp_objects += temp_containers
    temp_containers = list_to_sorts(temp_containers, 'containers')
    temp_containers += ' + #plates' if temp_plates else ''
    temp_containers += ' + #glasses' if temp_glasses else ''
    sorts.append(temp_containers+'.')
    others = sort_dict['others']
    temp_others = [item for item in others if item in items]
    if not temp_others:
        temp_others = [others[0]]
    temp_objects += temp_others
    if temp_others:
        temp_others = list_to_sorts(temp_others, 'others')
        sorts.append(temp_others+'.')
    # graspable
    variables = [('#food',temp_food), ('#drinks',temp_drinks), ('#containers',temp_containers), ('#others',temp_others)]
    temp_graspable = join_sorts(variables)
    if temp_graspable:
        sorts.append('#graspable = '+temp_graspable+'.')
        sorts.append('#objects = #appliances + #graspable.')
    else:
        sorts.append('#objects = #appliances.')
    sorts.append(list_to_sorts(locations,'furniture_surfaces')+'.')
    sorts.append('#surfaces = #furniture_surfaces + #appliances.')
    return temp_objects, sorts

# return filterd fluents according to sorts
def filter_fluents(fluents, temp_objects):
    all_fluents = []
    for fluent in fluents:
        if fluent.startswith('occurs'): # exo actions
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            items = [item.split(',')[1] if ',' in item else item for item in items]
            if items[0] in temp_objects:
                all_fluents.append(fluent)
        else:
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            items = [item.split(',')[0] if ',' in item else item for item in items]
            if items[0] in temp_objects or items[0] == 'human' or items[0] == 'ahagent1' or items[0] == 'ahagent2' or items[0] == 'ahagent3': # at
                if items[0] == 'human' or items[0] == 'ahagent1' or items[0] == 'ahagent2' or items[0] == 'ahagent3':
                    in_items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
                    in_items = [item.split(',')[1] if ',' in item else item for item in in_items]
                    sort_dict = get_sortlist()
                    locations = sort_dict['furniture_surfaces']
                    if in_items[0] in temp_objects or in_items[0] in locations:
                        all_fluents.append(fluent)
                else:
                    all_fluents.append(fluent)
    return all_fluents

def remove_unrelated_literals(ASP_goal, ah_fluents, common_fluents):
    sort_list = get_sortlist()
    objects, new_goal, new_fluents, new_comm_fluents = [], [], [], []
    for key, value in sort_list.items():
        objects.extend(value)
    objects = objects + ['human', 'ahagent1', 'ahagent2', 'agent']
    # ASP_goal
    literals = re.findall(r'holds\(.*?\),I\)', ASP_goal)
    for lit in literals:
        item = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',lit)][0]
        sub_items = []
        if ',' in item:
            sub_items.append(item.split(',')[0])
            sub_items.append(item.split(',')[1])
        else:
            sub_items.append(item)
        if set(sub_items).issubset(objects):
            new_goal.append(lit)
    ASP_goal = ', '.join(new_goal)

    # ah_fluents
    for fluent in ah_fluents:
        item = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)][0]
        sub_items = []
        if ',' in item:
            sub_items.append(item.split(',')[0])
            sub_items.append(item.split(',')[1])
        else:
            sub_items.append(item)
        if set(sub_items).issubset(objects):
            new_fluents.append(fluent)

    # common_fluents
    for fluent in common_fluents:
        item = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)][0]
        sub_items = []
        if ',' in item:
            sub_items.append(item.split(',')[0])
            sub_items.append(item.split(',')[1])
        else:
            sub_items.append(item)
        if set(sub_items).issubset(objects):
            new_comm_fluents.append(fluent)

    return ASP_goal, new_fluents, new_comm_fluents
    
# return answer sets for the new ASP file
def run_ASP_ahagent(graph, current_task, ASP_goal, const_timeout, all_prev_tasks, flags, ah_fluents, common_fluents, all_prev_actions, env_id, id_dict, current_script, comm_action, comm_exec, last_task, num_agents, current_agent_id, ah_at_dict, status_switchon_dict):
    if comm_action:
        update_action_axioms(comm_action, comm_exec)

    # modify the ASP_goal and fluents to only include known literals and sorts
    ASP_goal, ah_fluents, common_fluents = remove_unrelated_literals(ASP_goal, ah_fluents, common_fluents)

    ah_asp_pre = 'ASP/ahagent_pre' + str(current_agent_id) + '.sp'
    ah_asp_new = 'ASP/ahagent' + str(current_agent_id) + '.sp'
    sub_goal_success = False
    found_solution = False
    # answer_split = None
    counter = 1
    max_counter = const_timeout[0]
    timeout = const_timeout[1]
    exit_counter = 0
    reader = open(ah_asp_pre, 'r')
    pre_asp = reader.read()
    reader.close()
    pre_asp_split = pre_asp.split('\n')
    display_marker_index = pre_asp_split.index(display_marker)
    # get future actions for other agents
    other_actions, graphs, _ = get_future_actions(graph, current_task, all_prev_tasks, flags, ah_fluents, common_fluents, ASP_goal, all_prev_actions, env_id, id_dict, current_script, num_agents, current_agent_id)
    if last_task:
        _, fluents = get_fluents(ASP_goal, other_actions, graphs, ah_fluents+common_fluents, id_dict, num_agents, current_agent_id)
    else:
        ASP_goal, fluents = get_fluents(ASP_goal, other_actions, graphs, ah_fluents+common_fluents, id_dict, num_agents, current_agent_id)
    temp_objects, sorts = get_sorts_objects(ASP_goal, fluents)
    
    sorts.append('#agent = {ahagent' + str(current_agent_id) + '}.')
    agent_names = []
    for agent_idx in range(num_agents):
        if agent_idx not in [0, current_agent_id]:
            agent_name = 'ahagent' + str(agent_idx)
            agent_names.append(agent_name)
    agent_name_sort = '#other_agents = {human, ' + ', '.join(agent_names) + '}.'
    sorts.append(agent_name_sort)
    
    fluents = filter_fluents(fluents, temp_objects)
    if ASP_goal:
        while (not found_solution):
            const_term = ['#const n = ' + str(counter) + '.']
            asp_split = const_term + sorts + pre_asp_split[:display_marker_index] + ['goal(I) :- ' + ASP_goal + '.'] + fluents + pre_asp_split[display_marker_index:]
            asp = '\n'.join(asp_split)
            f1 = open(ah_asp_new, 'w')
            f1.write(asp)
            f1.close()
            try:
                plan_time_start = time.time()
                answer = subprocess.check_output('java -jar ASP/sparc.jar ' +ah_asp_new+' -A -n 1',shell=True, timeout=timeout)
                plan_time_end = time.time()
            except subprocess.TimeoutExpired as exec:
                print('command timed out')
                if counter <= max_counter:
                    counter = counter+1
                else:
                    if exit_counter < 2:
                        exit_counter = exit_counter+1
                        counter = 1
                        ASP_goal = remove_excess_fluents(ASP_goal)
                        continue
                    else:
                        print('reached MAX count!')
                        return None, False, 0
            answer_split = (answer.decode('ascii'))
            if len(answer_split) > 1:
                found_solution = True
                if len(answer_split) <= 3:
                    sub_goal_success = True
            else: # no answer exit const not large enough for an answer for this ASP
                if counter < max_counter:
                    counter = counter+1
                else:
                    if exit_counter <= 2:
                        exit_counter = exit_counter+1
                        counter = 1
                        ASP_goal = remove_excess_fluents(ASP_goal)
                        continue
                    else:
                        print('reached MAX count!')
                        return None, False, 0
    else:
        return None, False, 0
    plan_time_inside = plan_time_end-plan_time_start
    actions, sub_goal_success = process_answerlist(answer_split, sub_goal_success)
    if actions:
        action_fail = check_action_feasibility(deepcopy(actions), graph, ah_at_dict, id_dict, status_switchon_dict)
        if action_fail:
            actions = 'Failed'
    return actions, sub_goal_success, plan_time_inside

def get_sortlist():
    lists_dict = {}
    file_ = 'asp_sorts_partial.txt'
    with open(file_, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.split('=',1)
                key = key.strip()
                value = value.strip()
                try:
                    lists_dict[key] = ast.literal_eval(value) # for normal lists
                except (SyntaxError, ValueError): # for list combinations
                    multiple_lists = value.split('+')
                    combined_list = []
                    for part in multiple_lists:
                        part = part.strip()
                        try:
                            temp_list = ast.literal_eval(part) # for normal lists
                            combined_list.extend(temp_list)
                        except (SyntaxError, ValueError): # for list combinations
                            if part in lists_dict:
                                combined_list.extend(lists_dict[part])
                            else:
                                print('Undefined nested list')
                    lists_dict[key] = combined_list
    return lists_dict

def get_num_hand_objects(graph, id_dict):
    ah2_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 3 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    hand_obj = []
    for item in ah2_object_ids: # objects in add agent hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                hand_obj.append(name_)
    return len(hand_obj)
    
def get_obj_location(object, graph, id_dict):
    sort_dict = get_sortlist()
    temp_locations = sort_dict['furniture_surfaces']

    ah2_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 3 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    hand_obj = []
    for item in ah2_object_ids: # objects in add agent hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                hand_obj.append(name_)

    if object in hand_obj:
        return 'grabbed'
    
    human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    ah1_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    other_hand_obj = []
    for item in human_object_ids: # objects in human hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                other_hand_obj.append(name_)
    for item in ah1_object_ids: # objects in ah agent hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                other_hand_obj.append(name_)
    if object in other_hand_obj:
        return 'other_hand'

    for location in temp_locations:
        location_id = id_dict[location]
        edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == location_id and edge['relation_type'] == 'ON']
        edges_inside = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == location_id and edge['relation_type'] == 'INSIDE']
        all_edges = edges_on if location == 'counter_three' else edges_on + list(set(edges_inside)-set(edges_on))
        for edge in all_edges:
            for key, value in id_dict.items():
                if value == edge and key != 'microwave':
                    name_ = key
                    if name_ == object:
                        return location

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['chair'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                if name_ == object:
                    return 'livingroom_desk'

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['sofa'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                if name_ == object:
                    return 'livingroom_coffeetable'

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['microwave'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                if name_ == object:
                    return 'kitchen_smalltable'

    edges_inside = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['dishwasher'] and edge['relation_type'] == 'INSIDE']
    for edge in edges_inside:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                if name_ == object:
                    return 'dishwasher'

    if object == 'bananas': # on fruit bowl on livingroom_coffeetable
        return 'livingroom_coffeetable'
    if object == 'plum': # on fruit bowl on livingroom_coffeetable
        return 'livingroom_coffeetable'
    if object == 'breadslice': # inside toaster
        return 'counter_three'
    if object == 'dishwasher': # dishwasher == counter_three
        return 'counter_three'
    if object == 'cellphone': # on living room AIR
        return 'livingroom_coffeetable'
    if object == 'cereal': # on living room floor
        return 'livingroom_desk'
    if object == 'coffeepot': # on living room floor
        return 'livingroom_desk'


def get_app_status(appliance, graph, id_dict, status_switchon_dict):
    status = []
    if appliance == 'dishwasher':
        dishwasher_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
        if 'CLOSED' in dishwasher_status:
            status.append('CLOSE')
        elif 'OPEN' in dishwasher_status:
            status.append('OPEN')
        if status_switchon_dict['dishwasher'] == 1:
            status.append('ON')
        else:
            if 'OFF' in dishwasher_status:
                status.append('OFF')
            elif 'ON' in dishwasher_status:
                status.append('ON')
        return status
    if appliance == 'computer':
        computer_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'computer'][0]
        if status_switchon_dict['computer'] == 1:
            status.append('ON')
        else:
            if 'OFF' in computer_status:
                status.append('OFF')
            elif 'ON' in computer_status:
                status.append('ON')
        return status
    if appliance == 'coffeemaker':
        coffeemaker_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'coffeemaker'][0]
        if status_switchon_dict['coffeemaker'] == 1:
            status.append('ON')
        else:
            if 'OFF' in coffeemaker_status:
                status.append('ON')
            elif 'OFF' in coffeemaker_status:
                status.append('ON')
        return status

# VirtualHome does not fail its action in below cases, thus need to implement this explicitly and integrate into the flow
def check_action_feasibility(ah_actions, graph, ah_at_dict, id_dict, status_switchon_dict):
    action = ah_actions[0]
    action_fail = False
    for key, value in ah_at_dict.items():
        if value == 1:
            agent_location = key
    
    action_split = re.findall(r'\w+',action)
    
    # -occurs(move(R,L),I) :- holds(at(R,L),I)
    if action_split[1] == 'move' and agent_location == action_split[3]:
        action_fail = True
        print('Agent is already in that location ', agent_location, action_split[3])
        return action_fail
    
    if action_split[1] == 'grab':
        obj_location = get_obj_location(action_split[3], graph, id_dict)
        # -occurs(grab(R,G),I):- not holds(at(R,L),I), holds(on(A,L),I), holds(on(G,A),I).
        if obj_location == 'dishwasher' and agent_location != 'counter_three':
            action_fail = True
            print('Object inside appliance. Agent is not at the same location as the appliance. ', agent_location, obj_location)
            return action_fail
        # -occurs(grab(R,G),I):- not holds(at(R,L),I), holds(on(G,L),I)
        if obj_location != agent_location:
            action_fail = True
            print('Agent is not at the same location as the object. ', agent_location, obj_location)
            return action_fail
        # -occurs(grab(R,G),I):- holds(grabbed(R,G),I).
        if obj_location == 'grabbed':
            action_fail = True
            print('Object already in hand of the agent. ', agent_location, obj_location)
            return action_fail
        # -occurs(grab(R,G),I):- holds(agent_hand(T,G),I)
        if obj_location == 'other_hand':
            action_fail = True
            print('Object in the hand of another agent. ', agent_location, obj_location)
            return action_fail

    if action_split[1] == 'put':
        # these names could be easily read through sort file; this is a temp solution
        electricals = ['dishwasher']
        appliances = ['computer', 'coffeemaker']
        obj_location = get_obj_location(action_split[3], graph, id_dict)
        # -occurs(put(R,G,S),I) :- not holds(grabbed(R,G),I)
        if obj_location != 'grabbed':
            action_fail = True
            print('Object is not in the hand of agent. ', agent_location, obj_location)
            return action_fail
        # -occurs(put(R,G,L),I) :- not holds(at(R,L),I)
        if (action_split[4] not in electricals) and agent_location != action_split[4]:
            action_fail = True
            print('Agent is not in the destination. ', agent_location, action_split[4])
            return action_fail
        # -occurs(put(R,G,A),I) :- not holds(at(R,L),I), holds(on(A,L),I)
        if action_split[4] in electricals and agent_location != 'counter_three': # if adding more electricals this needs to be changed
            action_fail = True
            print('Agent is not in the destination ', agent_location, 'dishwasher')
            return action_fail
        # -occurs(put(R,G,E),I) :- not holds(opened(E),I), #agent(R)
        elec_status = get_app_status(action_split[4], graph, id_dict, status_switchon_dict)
        if (action_split[4] in electricals) and ('OPEN' not in elec_status):
            action_fail = True
            print('Appliance door is closed ', action_split[4])
            return action_fail

    if action_split[1] == 'switchon':
        num_objects = get_num_hand_objects(graph, id_dict)
        # -occurs(switchon(R,A),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2.
        if num_objects >= 2:
            action_fail = True
            print('Both hands are occupied as the agent is holding two objects.')
            return action_fail
        # -occurs(switchon(R,A),I):- not holds(at(R,L),I), holds(on(A,L),I).
        app_location = get_obj_location(action_split[3], graph, id_dict)
        if agent_location != app_location:
            action_fail = True
            print('Agent is not in the same loaction as the appliance ', action_split[3], app_location, agent_location)
            return action_fail
        # -occurs(switchon(R,E),I) :- holds(opened(E),I), #agent(R), #electricals(E).
        app_status = get_app_status(action_split[3], graph, id_dict, status_switchon_dict)
        if 'OPEN' in app_status:
            action_fail = True
            print('Appliance door is open ', action_split[3])
            return action_fail
        # -occurs(switchon(R,A),I):- holds(switchedon(A),I)
        if 'ON' in app_status:
            action_fail = True
            print('Appliance is already switched on ', action_split[3])
            return action_fail
        
    if action_split[1] == 'switchoff':
        num_objects = get_num_hand_objects(graph, id_dict)
        # -occurs(switchoff(R,A),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2
        if num_objects >= 2:
            action_fail = True
            print('Both hands are occupied as the agent is holding two objects.')
            return action_fail
        # -occurs(switchoff(R,A),I):- not holds(at(R,L),I), holds(on(A,L),I)
        app_location = get_obj_location(action_split[3], graph, id_dict)
        if agent_location != app_location:
            action_fail = True
            print('Agent is not in the same loaction as the appliance ', action_split[3], app_location, agent_location)
            return action_fail
        app_status = get_app_status(action_split[3], graph, id_dict, status_switchon_dict)
        # -occurs(switchoff(R,A),I):- not holds(switchedon(A),I)
        if 'OFF' in app_status:
            action_fail = True
            print('Appliance is already switched off ', action_split[3])
            return action_fail
        
    if action_split[1] == 'open':
        num_objects = get_num_hand_objects(graph, id_dict)
        # -occurs(open(R,E),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2
        if num_objects >= 2:
            action_fail = True
            print('Both hands are occupied as the agent is holding two objects.')
            return action_fail
        # -occurs(open(R,E),I):- not holds(at(R,L),I), holds(on(E,L),I)
        app_location = get_obj_location(action_split[3], graph, id_dict)
        if agent_location != app_location:
            action_fail = True
            print('Agent is not in the same loaction as the appliance ', action_split[3], app_location, agent_location)
            return action_fail
        app_status = get_app_status(action_split[3], graph, id_dict, status_switchon_dict)
        # -occurs(open(R,E),I):- holds(opened(E),I)
        if 'OPEN' in app_status:
            action_fail = True
            print('Appliance is already opened ', action_split[3])
            return action_fail
        # -occurs(open(R,E),I) :- holds(switchedon(E),I), #agent(R), #electricals(E).
        if 'ON' in app_status:
            action_fail = True
            print('Appliance is switched on. Switch off to open ', action_split[3])
            return action_fail
    if action_split[1] == 'close':
        num_objects = get_num_hand_objects(graph, id_dict)
        # -occurs(close(R,E),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2
        if num_objects >= 2:
            action_fail = True
            print('Both hands are occupied as the agent is holding two objects.')
            return action_fail
        # -occurs(close(R,E),I):- not holds(at(R,L),I), holds(on(E,L),I)
        app_location = get_obj_location(action_split[3], graph, id_dict)
        if agent_location != app_location:
            action_fail = True
            print('Agent is not in the same loaction as the appliance ', action_split[3], app_location, agent_location)
            return action_fail
        app_status = get_app_status(action_split[3], graph, id_dict, status_switchon_dict)
        # -occurs(close(R,E),I):- not holds(opened(E),I), #agent(R), #electricals(E).
        if 'CLOSE' in app_status:
            action_fail = True
            print('Appliance is already closed ', action_split[3])
            return action_fail
    return action_fail