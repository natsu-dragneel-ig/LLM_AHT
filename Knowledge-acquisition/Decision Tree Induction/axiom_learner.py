from __future__ import division
import os
import time
import re
import random
import csv
import numpy as np
import string
import collections
from shutil import copyfile
# for tree induction...
import pandas as pd
import math
import itertools
#from random import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sys

# This function create a initial state and an action for simulations. 
def randomState_Action(action, graspables, electricals, appliances, surfaces, furniture_surfaces, sub_locations): # for each action you need to write the grounding
	actionGround = 'none'
	initialState = []
	app_loc_dict ={
		'dishwasher': 'counter_three',
		'computer': 'livingroom_desk',
		'coffeemaker': 'counter_one'
	}
	
	if action == 'open':
		new_electrical = random.choice(electricals)
		actionGround = 'occurs(open(ahagent,'+new_electrical+'),0).'
		switchedon = np.random.choice([1,0])
		if switchedon:
			initialState.append('holds(switchedon('+new_electrical+'),0).')
		else:
			initialState.append('-holds(switchedon('+new_electrical+'),0).')
		if switchedon:
			opened = 0
		else:
			opened = np.random.choice([1,0])
		if opened:
			initialState.append('holds(opened('+new_electrical+'),0).')
		else:
			initialState.append('-holds(opened('+new_electrical+'),0).')
		elec_location = app_loc_dict[new_electrical]
		initialState.append('holds(on(' + new_electrical + ',' + elec_location + '),0).')
		if opened:
			same_location = 1
		else:
			same_location = np.random.choice([1,0])
		if same_location:
			initialState.append('holds(at(ahagent,'+elec_location+'),0).')
		else:
			furniture_surfaces.remove(elec_location)
			agent_location = random.choice(furniture_surfaces)
			initialState.append('holds(at(ahagent,'+agent_location+'),0).')
	
	if action == 'grab':
		new_graspable = random.choice(graspables) # sub_graspables
		actionGround = 'occurs(grab(ahagent,'+new_graspable+'),0).' # set the action as the initial step act
		inhand = 1 # np.random.choice([1,0])
		if inhand:
			initialState.append('holds(grabbed(ahagent,'+new_graspable+'),0).')
		else:
			# obj_location = np.random.choice(surfaces)
			obj_location = np.random.choice(electricals)
			initialState.append('holds(on(' + new_graspable + ',' + obj_location + '),0).')
			if obj_location in electricals:
				app_opened = np.random.choice([1,0])
				if app_opened:
					initialState.append('holds(opened(' + obj_location + '),0).')
				else:
					initialState.append('-holds(opened(' + obj_location + '),0).')
				elec_location = app_loc_dict[obj_location]
				initialState.append('holds(on(' + obj_location + ',' + elec_location + '),0).')
				obj_location = elec_location
			same_location = np.random.choice([1,0])
			if same_location:
				initialState.append('holds(at(ahagent,'+obj_location+'),0).')
			else:
				# furniture_surfaces.remove(obj_location)
				agent_location = random.choice(furniture_surfaces) # sub_locations
	# 			initialState.append('holds(at(ahagent,'+agent_location+'),0).')
	if action == 'move':
		new_lcoation = np.random.choice(furniture_surfaces)
		actionGround = 'occurs(move(ahagent,'+new_lcoation+'),0).' # set the action as the initial step act
		same_location = np.random.choice([1,0])
		if same_location:
			initialState.append('holds(at(ahagent,'+new_lcoation+'),0).')
		else:
			agent_location = random.choice(furniture_surfaces)
			initialState.append('holds(at(ahagent,'+agent_location+'),0).')

	if action == 'switchon':
		new_appliance = np.random.choice(electricals) #appliances
		actionGround = 'occurs(switchon(ahagent,'+new_appliance+'),0).' # set the action as the initial step act
		obj_location = app_loc_dict[new_appliance]
		initialState.append('holds(on(' + new_appliance + ',' + obj_location + '),0).')
		switchedon = np.random.choice([1,0])
		if new_appliance in electricals:
			opened = np.random.choice([1,0])
			if opened:
				initialState.append('holds(opened('+new_appliance+'),0).')
				switchedon = 0
			else:
				initialState.append('-holds(opened('+new_appliance+'),0).')
		same_location = np.random.choice([1,0])
		if same_location:
			initialState.append('holds(at(ahagent,'+obj_location+'),0).')
		else:
			agent_location = random.choice(furniture_surfaces)
			initialState.append('holds(at(ahagent,'+agent_location+'),0).')
		if switchedon:
			initialState.append('holds(switchedon('+new_appliance+'),0).')
		else:
			initialState.append('-holds(switchedon('+new_appliance+'),0).')
	#  - occurs(put(R,G,L),I) :- not holds(at(R,L),I),'
	# if action == 'put':
	# 	new_graspable = random.choice(graspables)
	# 	new_location = random.choice(furniture_surfaces)
	# 	actionGround = 'occurs(put(ahagent,'+new_graspable+','+new_location+'),0).' # set the action as the initial step act
	# 	inhand = 0.8
	# 	inhand = np.random.choice([1,0], p=[inhand,1-inhand])
	# 	if inhand:
	# 		initialState.append('holds(grabbed(ahagent,'+new_graspable+'),0).')
	# 		graspables.remove(new_graspable)
	# 	else:
	# 		obj_location = np.random.choice(furniture_surfaces)
	# 		initialState.append('holds(on(' + new_graspable + ',' + obj_location + '),0).')
	# 	in_loc = 0.2
	# 	same_location = np.random.choice([1,0], p=[in_loc,1-in_loc])
	# 	if same_location:
	# 		initialState.append('holds(at(ahagent,'+new_location+'),0).')
	# 	else:
	# 		agent_location = random.choice(furniture_surfaces)
	# 		initialState.append('holds(at(ahagent,'+agent_location+'),0).')
	return initialState, actionGround # return (grounded)state literal and the grounded action

# This function writes learned axioms to a file.
def writer(path, source):
    with open(path,"r") as base:
        agent_base_w = [line for line in base]
        
    with open(source,"a+") as learning:
        learned_axioms = [line for line in learning]

    agent_edited_with = open('temp.sp','w')
    for i, line in enumerate(agent_base_w):
        agent_edited_with.write(line)
        # write the initial state and the action in the ASP program
        if i == [n+2 for n, ax in enumerate(agent_base_w) if '%% Learned Rules' in agent_base_w[n]][0]:
            for axiom in learned_axioms:
                agent_edited_with.write(axiom)       
    agent_edited_with.close()
    copyfile('temp.sp', path)

# This function run an ASP program, including the action and initial configuration, and save the answer sets to a file.
# path=file name, action = grounded action, initialstate = grounded literal, answerfile = write the answer set to this file
def planner(path, action, initialState, answerfile):
    with open(path,"r") as base:
        agent_base_w = [line for line in base]

    agent_edited_with = open('temp_with.sp','w')
    for i, line in enumerate(agent_base_w):
        agent_edited_with.write(line)
        # write the initial state and the action in the ASP program
        if i == [n+2 for n, ax in enumerate(agent_base_w) if '%%Initial State:' in agent_base_w[n]][0]:
            for state in initialState:
                agent_edited_with.write(state+'\n')       
            agent_edited_with.write(action+'\n')
    agent_edited_with.close()
    #running the program and saving the answer set
    start = time.time()
    os.system("java -jar sparc.jar temp_with.sp -A > " + answerfile) # write to the file
    end = time.time()
    planTime = end - start # calculating planning time
	
    return planTime # get the answer set for the particular program with the groundeed initial state => not used -_-

# find literals in the Answer Set totally, partially or not grounded
# INPUTS: Expression -> partially/totally/non-grounded literal in string format
#         AnswerSetFile -> the file containing the Answer Set obtaioned from SPARC
# OUTPUT: list of matched grounded terms retrieved from the Answer Set
def AnswerSetFinder(Expression, AnswerSetFile):
    #openning Answer set file
    f = open(AnswerSetFile,"r")
    AnswerSet = f.read()
    f.close()
    if not re.search('[\>\<\=\!]',Expression):
        Expression = re.sub('\(', '\(', Expression)
        Expression = re.sub('\)', '\)', Expression)
        Expression = re.sub('[A-Z][0-9]?', '\\\S+(?:\(\\\S+?\))?', Expression)#'[a-z0-9_]+(?:\(.+?\))?', Expression)
        literal = re.findall('(?<!-)'+Expression, AnswerSet)
    else:
        literal = [Expression]
    return literal
    
# This function compares two answer sets and output the differences
# the ad hoc agent only have access to the answer set literals. Not the oracle ASP file (ground truth).
def compareAnsSet(AnswerSetExp, AnswerSetObs):
    # retrieves Expectation and observation from an Answer set
    expectation = AnswerSetFinder('holds(F,1)', AnswerSetExp)
    observation = AnswerSetFinder('holds(F,1)', AnswerSetObs)
    # comparing Expectation X Observation
    unexpected = [i for i in observation if i not in expectation]
    extra = [j for j in expectation if j not in observation]
    for ext in extra:
        inertialExtra = re.sub(',\s?1', ',0', ext)
        if (inertialExtra in AnswerSetFinder('holds(F,0)', AnswerSetExp)) and (os.lstat(AnswerSetObs).st_size > 5):
            unexpected.append('-'+ext)
            extra.remove(ext)
    return unexpected, extra

# Replacing ground terms for convenient variables   
def relRepresentation(grndLiterals, AllTerms, AllVar):
	# grndLiterals = [, [[], [actionGround], relevantFacts, 0], [[], [actionGround], relevantFacts, 0] ...
	#### "groundLiterals" is a list containing multiple lists in the format [inconsistence, action, relevantFluents, Label] with grounded terms
	AllRelevantFacts = []
	ungroundedLiterals = []
	for groundedLiterals in grndLiterals: # [[], [actionGround], relevantFacts, 0]
		flat_grndLit = [i for j in groundedLiterals[:-1] for i in j]
		for grndLit in flat_grndLit:
			term = grndLit[grndLit.find("(")+1:grndLit.rfind(")")]
			term = term[term.find("(")+1:term.rfind(")")]
			terms = term.split(',')
			var = [None]*len(terms)
			expression = grndLit
			expression = re.sub('\(', '\(', expression)
			expression = re.sub('\)', '\)', expression)
			# 1. Checking if a given term was replaced by a variable already 
			for i, term in enumerate(terms):
				if term in AllTerms:
					var[i] = AllVar[AllTerms.index(term)]
					expression = re.sub(term, var[i], expression)
				else:
					expression = re.sub(term, '[A-Z][0-9]?', expression)
					expression = re.sub(',\s?[0-9]', ',[0-9]', expression)
					# 2. for terms not matching any existing variable in the "AllRelevantFacts" in #1, replace terms by variables by looking at sorts.  
					var[i] = [key for key, value in sorts_dict.items() if term in value][0]
					seq = [r[0] for r in [re.findall('(?<='+var[i]+')[0-9]', variable) for variable in list(set(AllVar) | set(var)) if variable] if r]
					if seq:
						nTerm = max([int(num) for num in seq]) + 1
					else:
						nTerm = 1
					var[i] = var[i] + str(nTerm)
			AllTerms.extend(terms)
			AllVar.extend(var)
		# 4. replace the grounded fluents/action in the "groundLiterals" vector by their ungrounded versions
		ungroundLiteral = []
		for n, literals in enumerate(groundedLiterals[:-1]):
			ungroundedLit = []
			for literal in literals:
				ungrdLit = literal
				for i, term in enumerate(AllTerms): # the ungrdLit is chnaged below and resturned back to the loop
					if term.isdigit():
						pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
						ungrdLit = re.sub(','+pattern, ','+AllVar[i], ungrdLit) # check whether you have term inside ungrdLit if yes replace with AllVar[i] -sub(pattern, replacement, stringa)
					else:
						pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
						ungrdLit = re.sub(pattern, AllVar[i], ungrdLit) # check whether you have term inside ungrdLit if yes replace with AllVar[i] -sub(pattern, replacement, stringa)
				ungroundedLit.append(ungrdLit)
			ungroundLiteral.append(ungroundedLit)
		# 5. include the ungrounded flutes in the "AllRelevantFacts" list if not there yet.
			if n == 2:
				AllRelevantFacts = list(set(AllRelevantFacts) | set (ungroundedLit))
		ungroundLiteral.append(groundedLiterals[-1])                
		ungroundedLiterals.append(ungroundLiteral)
	return ungroundedLiterals, AllRelevantFacts, AllTerms, AllVar

# Retrieving the relevant information related to an action and store the terms involved
# actionGround, 'AnswerSetExp', relevantTerms = [] initially after that updated in each iteration in the train_ampunt loop = 100
def relevantInfo(action, answerSetFile, relevantTerms):
    #openning Answer set file
    f = open(answerSetFile,"r")
    answerSet = f.read()
    f.close()
    # identifying the grounded terms in the action
    # terms = action[action.find("(")+1:action.rfind(")")]
    # terms = terms[terms.find("(")+1:terms.rfind(")")]
    # actionTerms = terms.split(',')
    actionTerms = re.findall('([a-z][a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\()', action)
    relevantTerms.extend([i for i in actionTerms if i not in relevantTerms])
    # retrieving all the fluents and constants that contains the relevant terms as argument
    relevantFacts = []
    for term in actionTerms:
        relevantFact = re.findall("(?:\{|\s)(\S*"+term+"?\S+\),0\))", answerSet) # consider the literals at step 0 since the action is executed at step 0.
        for fact in relevantFact:
            terms = [t for t in re.findall('([a-z][a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\()', fact)]
            relevantTerms.extend([i for i in terms if i not in relevantTerms])
        relevantFacts = list(set(relevantFacts) | set(relevantFact)) # union of 2 lists without repetition
    return relevantFacts, relevantTerms


##########  TREE INDCUTION RELATED FUNCTIONS ############################################################

# FILTERING CANDIDATES
# input = X_test (test dataset), candidates (nodes of the selcted candidates), tests (features), labels (class labels,), thredholds (decision tree thresholds), level (0.95).
def filter_cands(X_test, candidates, tests, labels, thresholds, level):
    occurrence = [0]*len(candidates) # [0, 0, 0, ...]
    negative_count = [0]*len(candidates) # [0, 0, 0, ...]
    candidates = [list(i) for i in candidates] # candidates list
    for sample_id in range(len(X_test)): # for each sample in X_test test dataset
        # updating candidates information counts
        for idx,cnd in enumerate(candidates): # for each candidate as cnd
            if [X_test.iloc[sample_id][feature_names[feature[k]]] for k in cnd] == tests[idx]: # - or not
				# if the sample go through the candidate branch increase the occirance otherwise increase the neg
                if labels[idx] == X_label.iloc[sample_id]:
                    occurrence[idx] += 1
                else:
                    negative_count[idx] +=1

    extra_cands = []
    extra_tests = []
    extra_thresholds = []
    extra_labels = []
    #ranking and removing candidades
    idx_del =[]
    for idx,cand in enumerate(candidates):
        if occurrence[idx]/(occurrence[idx] + negative_count[idx] + 0.0001) < level:
            idx_del.append(idx)
        # comparing and deleting similar nodes in the same candidate, e.g., x>2 and x>3 results in the more specific x>3 
        if len(cand) > 1:
            cand_del = []
            for a, b in itertools.combinations(cand, 2):
                if feature[a] == feature[b]:
                    if (tests[idx][cand.index(a)] > thresholds[idx][cand.index(a)]) and (tests[idx][cand.index(b)] > thresholds[idx][cand.index(b)]):
                        if thresholds[idx][cand.index(a)] > thresholds[idx][cand.index(b)]:
                            cand_del.append(cand.index(b))
                        else:
                            cand_del.append(cand.index(a))
                    if (tests[idx][cand.index(a)] < thresholds[idx][cand.index(a)]) and (tests[idx][cand.index(b)] < thresholds[idx][cand.index(b)]):
                        if thresholds[idx][cand.index(a)] > thresholds[idx][cand.index(b)]:
                            cand_del.append(cand.index(a))
                        else:
                            cand_del.append(cand.index(b))
            candidates[idx] = [cand[i] for i in range(len(cand)) if i not in cand_del]
            tests[idx] = [tests[idx][i] for i in range(len(cand)) if i not in cand_del]
            thresholds[idx] = [thresholds[idx][i] for i in range(len(cand)) if i not in cand_del]
    #removing candidates that elaborates higher ranked
    for idx,cand in enumerate(candidates):
        for idx2,cand2 in enumerate(candidates):
            if (idx != idx2):
                if (idx not in idx_del) and (idx2 not in idx_del):
                    # deleting candidates that only elaborates higher ranked candidates
                    if all([feature[w] in [feature[f] for f in cand] for w in cand2]) and (occurrence[idx] <= occurrence[idx2]) and (labels[idx] == labels[idx2]):    
                        if ([thresholds[idx][[feature[h] for h in cand].index(feature[t])] for t in cand2] == thresholds[idx2]):
                            if ([tests[idx][[feature[h] for h in cand].index(feature[t])] for t in cand2] == tests[idx2]):
                                idx_del.append(idx)
                        else:
                            for idc,item in enumerate(cand2):
                                if (tests[idx][[feature[e] for e in cand].index(feature[item])] > thresholds[idx][[feature[e] for e in cand].index(feature[item])]) and (tests[idx2][idc] > thresholds[idx2][idc]):
                                    if (thresholds[idx][[feature[e] for e in cand].index(feature[item])]) > thresholds[idx2][idc]:                                 
                                        candidates[idx2][idc] = candidates[idx][[feature[e] for e in cand].index(feature[item])]
                                        thresholds[idx2][idc] = thresholds[idx][[feature[e] for e in cand].index(feature[item])]
                                        tests[idx2][idc] = tests[idx][[feature[e] for e in cand].index(feature[item])]
                                if (tests[idx][[feature[e] for e in cand].index(feature[item])] < thresholds[idx][[feature[e] for e in cand].index(feature[item])]) and (tests[idx2][idc] < thresholds[idx2][idc]):
                                    if (thresholds[idx][[feature[e] for e in cand].index(feature[item])]) < thresholds[idx2][idc]: 
                                        candidates[idx2][idc] = candidates[idx][[feature[e] for e in cand].index(feature[item])]
                                        thresholds[idx2][idc] = thresholds[idx][[feature[e] for e in cand].index(feature[item])]
                                        tests[idx2][idc] = tests[idx][[feature[e] for e in cand].index(feature[item])]
                            idx_del.append(idx)           

                    # creating possibly more general candidates from other two more elaborated
                    if any([item in cand for item in cand2]) and (labels[idx] == labels[idx2]):
                        if any([tests[idx][cand.index(t)]==tests[idx2][cand2.index(t)] for t in set(cand).intersection(cand2)]):
                            new_cand = []
                            new_test = []
                            new_thresh = []
                            for elem in [item for item in cand if item in cand2]:
                                if tests[idx][cand.index(elem)]==tests[idx2][cand2.index(elem)]:
                                    new_cand.append(elem)
                                    new_test.append(tests[idx][cand.index(elem)])
                                    new_thresh.append(thresholds[idx][cand.index(elem)])
                            if not(new_cand == cand or new_cand == cand2 or new_cand in extra_cands):
                                extra_cands.append(new_cand)
                                extra_tests.append(new_test)
                                extra_thresholds.append(new_thresh)
                                extra_labels.append(labels[idx])

    for remov in sorted(idx_del, reverse = True):
        del candidates[remov]
        del tests[remov]
        del labels[remov]
        del thresholds[remov]

    return(candidates, tests, labels, thresholds, extra_cands, extra_tests, extra_labels, extra_thresholds)

# CHECKING IF TWO UNGROUNDED AXIOMS ARE EQUAL OR A GENERAL\EXTENDED VERSION ONE ANOTHER REGARDLESS THE ORDER OF ITS LITERALS 
def compare_axioms(axiom1, axiom2):
    # correct_axiom = '-occurs(pass(R,O),I):-holds(has_ball(O),I).'
    # missing_axiom = '-occurs(pass(R,O),I):--holds(has_ball(R),I).'
    # if axiom1 == correct_axiom and axiom2 == missing_axiom:
    #     return [1, 1]
    # if  axiom1 == missing_axiom and axiom2 == correct_axiom:
    #     return [1, 1]
    literals1 = [lit.strip() for lit in re.split(':-|:\+|,(?![^(]*\))', axiom1.strip('.'))]
    expressions1 = [lit.strip() for lit in re.split(':-|:\+|,(?![^(]*\))', re.sub('\+','\+',re.sub('[A-Z][0-9]?', '[A-Z][0-9]?', re.sub('\)','\)',re.sub('\(','\(', axiom1.strip('.'))))))]
    literals2 = [lit.strip() for lit in re.split(':-|:\+|,(?![^(]*\))', axiom2.strip('.'))]
    expressions2 = [lit.strip() for lit in re.split(':-|:\+|,(?![^(]*\))', re.sub('\+','\+',re.sub('[A-Z][0-9]?', '[A-Z][0-9]?', re.sub('\)','\)',re.sub('\(','\(', axiom2.strip('.'))))))]
    intersection = [key for key in expressions1 if key in expressions2]
    var1 = []
    var2 = []
    for lit_expression in intersection:
        var1.extend([re.findall('[A-Z][0-9]?', literal) for literal in re.findall(lit_expression, axiom1)][0])
        var2.extend([re.findall('[A-Z][0-9]?', literal) for literal in re.findall(lit_expression, axiom2)][0])
             
    i1 = [[i for i, v1 in enumerate(var1) if var1[i] == a] for a in var1]
    i2 = [[i for i, v2 in enumerate(var2) if var2[i] == a] for a in var2]
    
    if i1 != i2:
        equal = 0
        version = 0
    else:
        equal = 1 if set(expressions1) == set(expressions2) else 0
        version = 0
        if expressions1[0] == expressions2[0]:
            version = 1 if ([n for n in expressions1 if n in expressions2] == expressions1) or ([n for n in expressions2 if n in expressions1] == expressions2) else 0
    # print(equal, version)
    return [equal, version]

def replace_variables(axioms):
	new_axioms = []
	for new_axiom in axioms:
		print('1', new_axiom)
		literals = re.findall(r'\(.*?\)',new_axiom)
		sub_literals = [(re.findall(r'\(.*?\)',literal[1:])) for literal in literals]
		variables = [sublit[0].strip('()') for sublit in sub_literals]
		variables = [var.split(',') for var in variables]
		variables = [element for var in variables for element in var]
		variables = list(set(variables))
		for key, value in sorts_dict.items():
			var_all = [var for var in variables if var[0] == key]
			if len(var_all) == 1:
				new_axiom = re.sub(var_all[0],key,new_axiom)
		# check whetehr a subsort of the vairables exisit in the header of the body of the axiom
		head_literal, body_literals = new_axiom.split(':-')
		head_variables = set(re.findall(r"\b[A-Z]\w*\b", head_literal))
		body_variables = set(re.findall(r"\b[A-Z]\w*\b", body_literals))
		head_only = head_variables - body_variables
		body_only = body_variables - head_variables
		if head_only and body_only:
			for head_var in head_only:
				if head_var in sorts_dict:
					var_list = set(sorts_dict[head_var])
					for key, value in sorts_dict.items():
						if key != head_var and key in body_only and var_list <= set(value):
							new_axiom = re.sub(rf'\b{head_var}\b', key, new_axiom)
			for body_var in body_only:
				if body_var in sorts_dict:
					var_list = set(sorts_dict[body_var])
					for key, value in sorts_dict.items():
						if key != body_var and key in head_only and var_list <= set(value):
							new_axiom = re.sub(rf'\b{body_var}\b', key, new_axiom)				
		new_axioms.append(new_axiom)
	return new_axioms

def filter_new_ax_final(new_axiom):
	literals = re.findall(r'\(.*?\)',new_axiom)
	sub_literals = [(re.findall(r'\(.*?\)',literal[1:])) for literal in literals]
	# sub_literals = [(re.findall(r'\(.*?\)',literals[0][1:])) for i in literals]
	variables = [sublit[0].strip('()') for sublit in sub_literals]
	variables = [var.split(',') for var in variables]
	all_variables = [var for var_list in variables for var in var_list]
	for varx in all_variables:
		if all_variables.count(varx) < 2:
			return None
	return new_axiom

# main...

#set of possible actions, objects and places

Agent = ['ahagent']
Food = ['bananas', 'cupcake']
Drinks = ['milk', 'juice']
Electricals = ['dishwasher']
Appliances = ['coffeemaker'] + Electricals
Plates = ['plate']
Glasses = ['waterglass']
Containers = ['coffeepot'] + Plates + Glasses
Others = ['book']
Furniture_Surfaces = ['kitchentable', 'livingroom_coffeetable', 'bedroom_coffeetable', 'livingroom_desk', 'bedroom_desk', 'counter_one', 'counter_three', 'kitchen_smalltable']
Sub_location = ['kitchentable', 'livingroom_coffeetable', 'bedroom_coffeetable']
Graspable =  Food + Drinks + Containers + Others
Surfaces = Furniture_Surfaces + Electricals
Objects = Appliances + Graspable
Actions = ['switchon'] #  'move', 'open', 'grab', 'close', 'put', 'switchon', 'switchoff'

sorts_dict = {
	'R': Agent, 'F': Food, 'D': Drinks,
	'E': Electricals, 'A': Appliances,
	'P': Plates, 'N': Glasses, 'C': Containers,
	'T': Others, 'L': Furniture_Surfaces, 'S': Surfaces, 
	'G':  Graspable, 'O': Objects
}

##########    EXPERIMENTS     ########

TPf = TPvf = FPf = FPvf = FNf = FNvf = 0 # confusion marix
falsePositive = []
falsePositiveV = []

for _ in range(1): # 20

###   Run different actions in different initial states ...

	copyfile('learner_missing.sp', 'agent_learner.sp')

	stop2 = 0 # iteration stop

	discovered_axioms = []

	while stop2 < 1: # 10
		unexpected = []
		extra = []

		stop1 = 0 # training stop
		while not unexpected and not extra and stop1 < 100: # 100 trains
			stop1 += 1
			electricals = Electricals[:]
			appliances = Appliances[:]
			graspables = Graspable[:]
			surfaces = Surfaces[:]
			furniture_surfaces = Furniture_Surfaces[:]
			sub_locations = Sub_location[:]

			action = random.choice(Actions) # trying out different actions to check if we will ever face a unexpected - inconsistency
			initialState, actionGround = randomState_Action(action, graspables, electricals, appliances, surfaces, furniture_surfaces, sub_locations)
			
			planner('learner_oracle.sp', actionGround, initialState, 'AnswerSetObs') # write the asnwer set from program  ASP_agent_oracle to AnswerSetExp
			planner('agent_learner.sp', actionGround, initialState, 'AnswerSetExp') # write the asnwer set from program  agent_learner to AnswerSetOb
		    # ... until find an inconsistency between observation and expectation:
			
			unexpected, extra = compareAnsSet('AnswerSetExp', 'AnswerSetObs')
		if stop1 >= 100: # 100 stop since no inconsistency found
			print('No inconsistencies found in the current knowledge.')
			break   
		### After finding any incompatibility, execute the same action for different arguments and initial states, and store the relevant information for the decision tree induction

		train_amount = 100

		count_unexp = 0
		count_extra = 0
		positive_extra = 0
		positive_unexp = 0
		causal_law = []
		execut_cond = []
		AllRelevantFacts = []
		relevantTerms = []
		unexpect_lit = []


		while count_unexp < train_amount and count_extra < train_amount: # for 100 times
			electricals = Electricals[:]
			appliances = Appliances[:]
			graspables = Graspable[:]
			surfaces = Surfaces[:]
			furniture_surfaces = Furniture_Surfaces[:]
			sub_locations = Sub_location[:]

			initialState, actionGround = randomState_Action(action, graspables, electricals, appliances, surfaces, furniture_surfaces, sub_locations)
			planner('learner_oracle.sp', actionGround, initialState, 'AnswerSetObs') # write the asnwer set from program  ASP_agent_oracle to AnswerSetExp(erved)
			planner('agent_learner.sp', actionGround, initialState, 'AnswerSetExp') # write the asnwer set from program  agent_learner to AnswerSetObs(ected)

			unexpected, extra = compareAnsSet('AnswerSetExp', 'AnswerSetObs') # find all unexpected and extra literals

			# print(actionGround)
			# print(initialState)
			# print(unexpected)
			# print(extra)
			# print('---------------------------------------------------------------')

			# finding the relevant facts
			relevantFacts, relevantTerms = relevantInfo(actionGround, 'AnswerSetExp', relevantTerms)  # relevantTerms here is incomplete - it needs to have the grid values

			# WHAT ABOUT THE CASES WHEN THERE ARE MULTIPLE INCONSISTENCIES?????      
			if unexpected and positive_unexp < 0.7*train_amount:
				causal_law.append([unexpected, [actionGround], relevantFacts, 1])
				count_unexp += 1
				positive_unexp += 1
				if count_extra < 0.3*train_amount or positive_extra >= 0.7*train_amount:
					execut_cond.append([[], [actionGround], relevantFacts, 0]) 
					count_extra += 1 
			      
			elif extra and positive_extra < 0.7*train_amount:
				if os.lstat('AnswerSetObs').st_size > 5: # size in bytes of the oracle file; amount of data waiting - usually 5400
					execut_cond.append([extra, [actionGround], relevantFacts, 1])
				else:
					execut_cond.append([[], [actionGround], relevantFacts, 1])
				count_extra += 1
				positive_extra += 1
			else:
				if count_unexp < 0.3*train_amount or positive_unexp >= 0.7*train_amount:
					causal_law.append([unexpected, [actionGround], relevantFacts, 0])
					count_unexp += 1
				if count_extra < 0.3*train_amount or positive_extra >= 0.7*train_amount:
					if os.lstat('AnswerSetObs').st_size > 5:    
						execut_cond.append([extra, [actionGround], relevantFacts, 0])
					else:
						execut_cond.append([[], [actionGround], relevantFacts, 0])
					count_extra += 1
		causal_input = []
		executability_input = []

		AllTerms = []
		AllVar = []
		if count_unexp == train_amount:
			causal_law, AllRelevantFacts, AllTerms, AllVar = relRepresentation(causal_law, AllTerms, AllVar)
			unexpect_lit = collections.Counter([causal[0][0] for causal in causal_law if causal[0]])
		    # sorting the list for saving in the csv file
			AllRelevantFacts.extend(['inconsitent'])
			causal_input.append(AllRelevantFacts)#.extend(['inconsitent'])) #creating the label column

			for causal_event in causal_law:
				new_line = []
				for fact in AllRelevantFacts[:-1]:
					if fact in causal_event[2]:
						new_line.append(1)
					else:
						new_line.append(0)
				new_line.append(causal_event[-1]) # fill in the label value
				causal_input.append(new_line)
		    # save causal law in a ".csv" file
			with open('treeInput.csv', 'w') as tree_input:
				wr = csv.writer(tree_input, quoting=csv.QUOTE_ALL)
				for causal_line in causal_input:
					wr.writerow(causal_line)
		    # saving the relevant data for axioms construction in the tree induction algorithm
			if len(unexpect_lit) > 1:
				var = re.findall('[A-Z][0-9]?', causal_law[-1][1][0])
				tot_vars = []
				for unexp in unexpect_lit.keys():
					tot_vars.append(sum([v in unexp for v in var])) 
				unexp_literal = list(unexpect_lit.keys())[tot_vars.index(max(tot_vars))]
			else:
				unexp_literal = list(unexpect_lit.keys())[0]
			info = ['causal_law', [unexp_literal], [causal_law[-1][1][0].replace('.','')], (len(AllRelevantFacts) - 1)]		    
		else:
			execut_cond, AllRelevantFacts, AllTerms, AllVar = relRepresentation(execut_cond, AllTerms, AllVar)
			# sorting the list for saving in the csv file
			AllRelevantFacts.extend(['inconsitent'])
			executability_input.append(AllRelevantFacts)
			for executability_event in execut_cond:
				new_line = []
				for fact in AllRelevantFacts[:-1]:
					if fact in executability_event[2]:
						new_line.append(1)
					else:
						new_line.append(0)
				new_line.append(executability_event[-1]) # fill in the label value
				executability_input.append(new_line)

		    # save causal law in a ".csv" file
			with open('treeInput.csv', 'w') as tree_input:
				wr = csv.writer(tree_input, quoting=csv.QUOTE_ALL)
				for execut_line in executability_input:
					wr.writerow(execut_line)

			info = ['executability_cond', [], [execut_cond[-1][1][0].replace('.','')], (len(AllRelevantFacts) - 1)]

		stop2 += 1
		######## TRAINING THE DECISION TREE ###############################################
		#importing the data from .CSV file
		dataset = pd.read_csv('treeInput.csv')
		n_features = info[-1]  # ex: info = ['executability_cond', [], ['occurs(shoot(R1,A1),0)'], 341] thereofre, n_features = 341
		test_size = 0.30 # 0.20
		n_trees = 50
		level = 0.90 # 0.98

		feature_names = dataset.columns.tolist()[:n_features] # hold statements in tree column names in csv file
		classes_names = dataset.columns.tolist()[n_features:] # ex : 'inconsitent'

		# count_lit = []
		new_axiom = []
		if len(feature_names) > 0:		
			literals = []
			count_lit = []
			negs = []
			threshols = []

			for idn, class_name in enumerate(classes_names):
				for it in range(n_trees):
					#formating the input
					train_features, X_test, train_targets, X_label = train_test_split(dataset.iloc[:,:n_features], dataset.iloc[:,n_features+idn], test_size=test_size)

					#inducting the decision tree
					tr = DecisionTreeClassifier(criterion = 'entropy',min_samples_split=15, min_samples_leaf=5).fit(train_features,train_targets)

					#exporting the tree
					tree.export_graphviz(tr,out_file='tree_'+class_name+'.dot',feature_names=feature_names,proportion=False,leaves_parallel=True,precision=2)#,class_names=['stable/non-occluded','stable/occluded','unstable/non-occluded','unstable/occluded'])

					#checking which features are binary or not
					feature_type = []
					for feature in feature_names:
						if max([int(f) for f in X_test.iloc[:][feature]]) > 1:
							feature_type.append('not')
						else:
							feature_type.append('bin')

					# The decision estimator has an attribute called tree_  which stores the entire
					# tree structure and allows access to low level attributes. The binary tree
					# tree_ is represented as a number of parallel arrays. The i-th element of each
					# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
					# Some of the arrays only apply to either leaves or split nodes, resp. In this
					# case the values of nodes of the other type are arbitrary!
					#
					# Among those arrays, we have:
					#   - left_child, id of the left child of the node
					#   - right_child, id of the right child of the node
					#   - feature, feature used for splitting the node
					#   - threshold, threshold value at the node
					#

					# Using those arrays, we can parse the tree structure:

					n_nodes = tr.tree_.node_count
					children_left = tr.tree_.children_left
					children_right = tr.tree_.children_right
					feature = tr.tree_.feature
					threshold = tr.tree_.threshold
					value = tr.tree_.value
					node_samples = tr.tree_.n_node_samples

					samples_tot = np.sum(value[0])
					print("Total samples = %s." % samples_tot)

					# The tree structure can be traversed to compute various properties such
					# as the depth of each node and whether or not it is a leaf.
					node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
					is_leaves = np.zeros(shape=n_nodes, dtype=bool)
					stack = [(0, -1)]  # seed is the root node id and its parent depth
					while len(stack) > 0:
						node_id, parent_depth = stack.pop()
						node_depth[node_id] = parent_depth + 1

						# If we have a test node
						if (children_left[node_id] != children_right[node_id]):
							stack.append((children_left[node_id], parent_depth + 1))
							stack.append((children_right[node_id], parent_depth + 1))
						else:
							is_leaves[node_id] = True

					print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)
					for i in range(n_nodes):
						if is_leaves[i]:
							print("%snode=%s leaf node, with number of samples = %s, and values = %s" % (node_depth[i] * "\t", i, node_samples[i], value[i]))
						else:
							print("%snode=%s test node: go to node %s if not %s else to node %s."
							% (node_depth[i] * "\t",
							i,
							children_left[i],
							feature_names[feature[i]],
							children_right[i],
							))
					print("End of tree %s" % idn)
          
					############## CREATING CANDIDATE AXIOMS:  ####################################


					# First let's retrieve the decision path of each sample. The decision_path
					# method allows to retrieve the node indicator functions. A non zero element of
					# indicator matrix at the position (i, j) indicates that the sample i goes
					# through the node j.

					node_indicator = tr.decision_path(X_test)

					# Similarly, we can also have the leaves ids reached by each sample.

					leave_id = tr.apply(X_test)

					# Now, it's possible to get the tests that were used to predict a sample or
					# a group of samples. Let's make it for all samples in the test data.

					att_branches = []
					candidates = []
					labels = []
					tests = []
					thresholds= []

					# building the candidates based on if they have at least one example suporting it and 
					# if the corresponding leaf agree (in some minimum percentage - 95% in this example) with its label.
					# Also if this leaf contains a representative percentage of training examples (e.g. 2%).

					for sample_id in range(len(X_test)):
						node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
										node_indicator.indptr[sample_id + 1]] # het all the nodes that sample_id went through
										
						new_branch = 0 # need to check this to learn axioms with multiple body literals
						if (list(node_index) not in [list(item) for item in att_branches]):
							new_branch = 1
							att_branches.append(node_index)

						print('Rules used to predict sample %s: %s' % (sample_id, node_index))
						for node_id in node_index:
							# continue to the next node if the agent reaches a leaf node
							if leave_id[sample_id] == node_id:
								print("leaf id node %s" % node_id)
								if ((np.max(value[node_id])/np.sum(value[node_id])) > level) and ((np.sum(value[node_id])/samples_tot) > 0.02):
									# if the conditions of at least 95% of samples suporting a class and minimum of 2% of the total in a leaf, the branch is considered for providing candidates
									if new_branch:
										# building candidates from this branch
										cands = (math.pow(2, len(node_index[:-1]))-1) #if len(node_index[:-1]) < 4 else math.pow(2, len(node_index[:-1])-1)
										# Number of candidates proportional to the number of nodes. With "n" being the number of nodes, for n = 1, 2, 3, cands = 2^n - 1 (number of possible combinations), otherwise cands = 2^(n-1).
										# ex: if nodes of sample_id = 1 => [0,1,3] then cands = 0, 1, 3, (0,1), (0,3), (1,3), (0,1,3)
										all_cands = []
										for j in range(len(node_index[:-1])):
											all_cands.extend(list(itertools.combinations(sorted(node_index[:-1]), j+1)))
										# all_cand = 0, 1, 3, (0,1), (0,3), (1,3), (0,1,3)
										for k in range(int(cands)):
											# drawing cadidates from the branch (excluding the leaf)
											idx_cand = np.random.choice(len(all_cands))
											candidate = all_cands[idx_cand]
											all_cands.remove(candidate)
											lab = np.argmax(value[node_id]) # het te dominnent class
											if (list(candidate) not in [list(item) for item in candidates]) and lab == 1: # if candidate not discovered before and the class is the dominnet class
												# if the candidate is not in the list, it is included:
												candidates.append(candidate) # -nodes in candidate
												labels.append(np.argmax(value[node_id])) # -labels supported by the leaf
												tests.append([X_test.iloc[sample_id][feature_names[feature[item]]] for item in candidate]) # -the test performed in each node (True or False)  
												thresholds.append([threshold[h] for h in candidate])
								continue
					candidates, tests, labels, thresholds, extra_cands, extra_tests, extra_labels, extra_thresholds = filter_cands(X_test, candidates, tests, labels, thresholds, level)     

					if len(extra_cands) > 0:
						candidates.extend(extra_cands)
						tests.extend(extra_tests)
						labels.extend(extra_labels)
						thresholds.extend(extra_thresholds)
						candidates, tests, labels, thresholds,extra_cands, extra_tests, extra_labels, extra_thresholds = filter_cands(X_test, candidates, tests, labels, thresholds, level)

					#printing the final results
					# labels = {1: "inconsistent", 0: "consistent"}
					# heads = {1: -'action': for executability condition; 2. 'unexpected fluent' (I + 1): for causal laws}
					# bodies = {1: precondition extract form decision tree: for executability condition; 2. 'action' (+ precondition from decision tree): for causal laws} 
					if info[0] == 'executability_cond': # info ex: ['executability_cond', [], ['occurs(shoot(R1,A1),0)'], 341]
						print("     EXECUTABILITY CONDITIONS:     ")
						for idx, candidate in enumerate(candidates):
							print(candidate)
							literal = []
							neg = []
							threshol = []
							sys.stdout.write("-")
							neg.append(0)
							sys.stdout.write(re.sub('\),\n?[0-9]\)','),I)',info[2][0]))
							literal.append(re.sub('\),\n?[0-9]\)','),I)',info[2][0]))
							threshol.append(0) # ?????????
							sys.stdout.write(":-")
							for idc,axiom in enumerate(candidate):
								print(axiom)
								literal.append(feature_names[feature[axiom]])
								if feature_type[feature[axiom]] == 'bin': # binary feature
									if tests[idx][idc] == 0:
										sys.stdout.write("-")
									neg.append(tests[idx][idc])
									print(tests[idx][idc])
									sys.stdout.write(re.sub('\),\n?[0-9]\)','),I)',feature_names[feature[axiom]]))
								#else:
								#    sys.stdout.write(feature_names[feature[axiom]])
								#    if tests[idx][idc] > thresholds[idx][idc]:
								#        sys.stdout.write(">")
								#        neg.append(1)
								#    else:
								#        sys.stdout.write("<=")
								#         neg.append(0)
								#    sys.stdout.write(str(int(thresholds[idx][idc])))
								threshol.append(int(thresholds[idx][idc])) ## ??????????
								if idc < (len(candidate)-1):
									sys.stdout.write(",")
							print(".")

							if literal in literals:
								print('if', literal)
								for i in [indx for indx, x in enumerate(literals) if x == literal]: 
									if neg == negs[i] and threshol == threshols[i]:
										count_lit[i] += 1
							else:
								print('else', literal)
								literals.append(literal)
								count_lit.append(1)
								negs.append(neg)
								threshols.append(threshol)
					else: # info[0] == 'causal_law'
						print("     CAUSAL LAWS:     ")
						for idx, candidate in enumerate(candidates):
							literal = []
							neg = []
							threshol = []
							sys.stdout.write(re.sub('\),\n?[0-9]\)','),I+1)',info[1][0]))
							literal.append(re.sub('\),\n?[0-9]\)','),I+1)',info[1][0]))
							neg.append(1)
							sys.stdout.write(":-")
							sys.stdout.write(re.sub('\),\n?[0-9]\)','),I)',info[2][0]))
							literal.append(re.sub('\),\n?[0-9]\)','),I)',info[2][0]))
							threshol.append(0) # ?????????
							neg.append(1)
							if not candidate:
								sys.stdout.write(".")
							else:
								sys.stdout.write(",")    
								for idc,axiom in enumerate(candidate):
									literal.append(feature_names[feature[axiom]])
									if feature_type[feature[axiom]] == 'bin':
										if tests[idx][idc] == 0:
											sys.stdout.write("-")
										neg.append(tests[idx][idc])
										sys.stdout.write(re.sub('\),\n?[0-9]\)','),I)',feature_names[feature[axiom]]))
									threshol.append(int(thresholds[idx][idc])) ## ??????????
									if idc < (len(candidate)-1):
										sys.stdout.write(",")
								print(".")
								if literal in literals: 
									for i in [indx for indx, x in enumerate(literals) if x == literal]: 
										if neg == negs[i] and threshol == threshols[i]:
											count_lit[i] += 1
								else:
									literals.append(literal)
									count_lit.append(1)
									negs.append(neg)
									threshols.append(threshol)		
		else: # no literals in the feature list
			literals = []

		f = open("axioms_%.2f.txt" % level, "w+")
		if not literals and info[0] == 'causal_law':
			new_ax = re.sub('\),\n?[0-9]\)','),I+1)',info[1][0])
			new_ax = new_ax + ':-'
			new_ax = new_ax + re.sub('\),\n?[0-9]\)','),I)',info[2][0])
			new_ax = new_ax + '.'
			new_ax = filter_new_ax_final(new_ax) # check this later
			if new_ax != None:
				new_axiom = [new_ax]

		#  need to check this
		if info[0] == 'executability_cond' and count_lit:
			n = max(count_lit)
		else:
			n = 0.4*n_trees

		for idx, candidate in enumerate(literals):
			if count_lit[idx] >= n:
				new_ax = ''
				if negs[idx][0] == 0: # for the action
					new_ax = new_ax + '-'
				new_ax = new_ax + candidate[0]
				new_ax = new_ax + ':-'
				for ind, prop in enumerate(candidate[1:]):
					if negs[idx][ind+1] == 0:
						if re.search(r'-holds', prop):
							prop = prop[1:]
						else:
							new_ax = new_ax + '-'
					new_ax = new_ax + re.sub('\),\n?[0-9]\)','),I)',prop)
					if ind < (len(candidate)-2):
						new_ax = new_ax + ','
				new_ax = new_ax + '.'
				new_axiom.append(new_ax)
		
		if new_axiom:
			new_axiom = replace_variables(new_axiom)
			f.write(new_axiom[0])
			discovered_axioms.extend(new_axiom)
			f.write('\n')
		f.close()		

	#########################################################################################################

		print(count_unexp)
		print(count_extra)
		print(stop1)
		print(stop2)

		print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print ('@                            #############                                                   @')
		print ('@                                #####                                                       @')
		print ('@                                #####                                                       @')
		print ('@                                #####                                                       @')
		print ('@                                #####                                                       @')
		print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

# print (float(TPf/(TPf+FPf))) # precision
# print (float(TPf/(TPf+FNf))) # recall
# print (float(TPvf/(TPvf+FPvf)))
# print (float(TPvf/(TPvf+FNvf)))
