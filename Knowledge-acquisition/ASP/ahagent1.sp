#const n = 2.
sorts
#food = {candybar, plum}.
#drinks = {coffee, coffee}.
#electricals = {dishwasher}.
#appliances = {coffeemaker} + #electricals.
#glasses = {mug}.
#containers = {coffeepot, coffeepot} + #glasses.
#graspable = #food + #drinks + #containers.
#objects = #appliances + #graspable.
#furniture_surfaces = {kitchentable, livingroom_coffeetable, bedroom_coffeetable, livingroom_desk, bedroom_desk, counter_one, counter_three, kitchen_smalltable}.
#surfaces = #furniture_surfaces + #appliances.
#agent = {ahagent1}.
#other_agents = {human, ahagent2}.
% #agent = {ahagent}.
% #other_agents = {human}.
#all_agents = #agent + #other_agents.
#step = 0..n.

#agent_actions = move(#agent,#furniture_surfaces) + grab(#agent,#graspable) + put(#agent,#graspable,#surfaces) + 
                 open(#agent,#electricals) + close(#agent,#electricals) + switchon(#agent,#appliances) + switchoff(#agent,#appliances).
#exogenous_actions = exo_move(#other_agents,#furniture_surfaces) + exo_grab(#other_agents,#graspable) + exo_put(#other_agents,#graspable,#surfaces) + exo_open(#other_agents,#electricals) + exo_close(#other_agents,#electricals) + exo_switchon(#other_agents,#appliances) + exo_switchoff(#other_agents,#appliances).
#actions = #agent_actions + #exogenous_actions.

#inertial_f = at(#agent,#furniture_surfaces) + grabbed(#agent,#graspable) + on(#objects,#surfaces) + opened(#electricals) + switchedon(#appliances) + made(#drinks) + agent_hand(#other_agents,#graspable).
#defined_f = agent_at(#other_agents,#furniture_surfaces).
#fluents = #inertial_f + #defined_f.

predicates
holds(#fluents, #step).
occurs(#actions, #step).
next_to(#furniture_surfaces, #furniture_surfaces).

%% planning
success().
something_happened(#step).
goal(#step).
planning(#step).
current_step(#step).

rules
% -------------------------- casual laws --------------------------%

% move causes the agent to be at that location. This includes walking and turning.
holds(at(R,L),I+1) :- occurs(move(R,L),I).

% grab causes an object to be in the hand of the agent.
holds(grabbed(R,G),I+1) :- occurs(grab(R,G),I).

% grab causes an object to be removed form the current surface.
-holds(on(G,S),I+1) :- occurs(grab(R,G),I), holds(on(G,S),I).

% put causes an object to be placed in the relevant surface.
holds(on(G,S),I+1) :- occurs(put(R,G,S),I), holds(grabbed(R,G),I), G != S.

% put causes an object to be removed from the agent hand.
-holds(grabbed(R,G),I+1) :- occurs(put(R,G,S),I), holds(grabbed(R,G),I), G != S.

% open causes the agent to open a door of a electrical.
holds(opened(E),I+1) :- occurs(open(R,E),I), not holds(opened(E),I).

% close causes the agent to close a door of a appliance.
-holds(opened(E),I+1) :- occurs(close(R,E),I), holds(opened(E),I).

% switch on causes the agent to switch on an appliance.
holds(switchedon(A),I+1) :- occurs(switchon(R,A),I), not holds(switchedon(A),I).

% switch off causes the agent to switch off an appliance.
-holds(switchedon(A),I+1) :- occurs(switchoff(R,A),I), holds(switchedon(A),I).

% exogeneous actions effect
holds(agent_at(O,L),I+1) :- occurs(exo_move(O,L),I), #other_agents(O), #furniture_surfaces(L).
holds(agent_hand(O,G),I+1) :- occurs(exo_grab(O,G),I), #other_agents(O), #graspable(G).
-holds(on(G,S),I+1) :- occurs(exo_grab(O,G),I), holds(on(G,S),I), #other_agents(O), #graspable(G), #surfaces(S).
holds(on(G,S),I+1) :- occurs(exo_put(O,G,S),I), #other_agents(O), #graspable(G), #surfaces(S).
-holds(agent_hand(O,G),I+1) :- occurs(exo_put(O,G,S),I), #other_agents(O), #graspable(G), #surfaces(S).
holds(opened(E),I+1) :- occurs(exo_open(O,E),I), #other_agents(O), #electricals(E).
-holds(opened(E),I+1) :- occurs(exo_close(O,E),I), #other_agents(O), #electricals(E).
holds(switchedon(A),I+1) :- occurs(exo_switchon(O,A),I), #other_agents(O), #appliances(A).
-holds(switchedon(A),I+1) :- occurs(exo_switchoff(O,A),I), #other_agents(O), #appliances(A).

% ----------------------- state constraints -----------------------%

% agent cannot be in two places at the same time.
-holds(at(R,L1),I) :- holds(at(R,L2),I), L1 != L2, #agent(R), #furniture_surfaces(L1), #furniture_surfaces(L2).

% coffee is made if the coffeepot is on the coffeemaker and the coffeemaker is switched_on.
holds(made(coffee),I) :- holds(on(coffeepot,coffeemaker),I), holds(switchedon(coffeemaker),I).

% prevent agent from starting unnecessary actions that will prolong the game time - need more like this
holds(made(coffee),I+1) :- holds(agent_hand(O,coffeepot),I), #other_agents(O).

% next_to works both ways. If not specified then not next_to each other
next_to(L1,L2) :- next_to(L2,L1).
-next_to(L1,L2) :- not next_to(L1,L2).

-holds(agent_hand(O,G),I+1) :- holds(agent_hand(O,G),I).

% -------------------- executability conditions -------------------%

%% move
% impossible to move to a location if the agent is already in that location.
-occurs(move(R,L),I) :- holds(at(R,L),I), #agent(R), #furniture_surfaces(L).

% impossible to move beteen furniture_surfaces that are not next to each other.
-occurs(move(R,L1),I) :- holds(at(R,L2),I), -next_to(L1,L2).

% cannot move to a different location if what is in your hand is supposed to be down in the previous location
-occurs(move(R,L1),I) :- holds(at(R,L),I), holds(grabbed(R,G),I), holds(on(G,L),I+I0), #step(I0), #agent(R), #furniture_surfaces(L), L != L1.

%% grab
% impossible to grab an object if the agent is not at the same location as the object.
-occurs(grab(R,G),I):- not holds(at(R,L),I), holds(on(G,L),I), #agent(R), #graspable(G), #furniture_surfaces(L).

% impossible to grab something from an appliance if the agent is not in the same location as the electrical.
-occurs(grab(R,G),I):- not holds(at(R,L),I), holds(on(A,L),I), holds(on(G,A),I), #agent(R), #graspable(G), #appliances(A), #furniture_surfaces(L).

% impossible to grab something from inside an electrical if the door is closed.
-occurs(grab(R,G),I):- not holds(opened(E),I), holds(on(G,E),I), #agent(R), #graspable(G), #electricals(E).

% impossible to grab something if that object is already in the hand of the agent.
-occurs(grab(R,G),I):- holds(grabbed(R,G),I), #agent(R), #graspable(G).

% impossible to grab something if that object is in the hand of the other agent.
-occurs(grab(R,G),I):- holds(agent_hand(T,G),I), #other_agents(T), #agent(R), #graspable(G).

% impossible to grab a third object if two objects are already in the hand of the agent
-occurs(grab(R,O3),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2, O2 != O3, O1 != O3, #agent(R), #graspable(O1), #graspable(O2), #graspable(O3).

% impossible to grab coffee - not required; but lead to error when the agent randomly decide to pick somehtng while waiting
-occurs(grab(R,coffee),I) :- #agent(R).

%% put
% impossible to put an object down if the objects is not in the hand of the agent.
-occurs(put(R,G,S),I) :- not holds(grabbed(R,G),I), #agent(R), #graspable(G), #surfaces(S).

% impossible to put an object down if the agent is not at the location.
-occurs(put(R,G,L),I) :- not holds(at(R,L),I), #agent(R), #graspable(G), #furniture_surfaces(L).

% impossible to put an object inside an appliance if the agent is not at the location of the appliance.
-occurs(put(R,G,A),I) :- not holds(at(R,L),I), holds(on(A,L),I), #agent(R), #graspable(G), #furniture_surfaces(L), #appliances(A).

% impossible to put something inside an electrical if the door is closed.
-occurs(put(R,G,E),I) :- not holds(opened(E),I), #agent(R), #graspable(G), #electricals(E).

%% switchon
% impossible to switchon if two objects are already in the hand of the agent
-occurs(switchon(R,A),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2, #agent(R), #graspable(O1), #graspable(O2), #appliances(A).

% impossible to switch on an appliance before finding it.
-occurs(switchon(R,A),I):- not holds(at(R,L),I), holds(on(A,L),I), #agent(R), #appliances(A), #furniture_surfaces(L).

% impossible to switch_on an electrical unless the door is closed.
-occurs(switchon(R,E),I) :- holds(opened(E),I), #agent(R), #electricals(E).

% impossible to switch_on an appliance if it is already switched_on.
-occurs(switchon(R,A),I):- holds(switchedon(A),I), #agent(R), #appliances(A).

%% switchoff
% impossible to switchoff if two objects are already in the hand of the agent
-occurs(switchoff(R,A),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2, #agent(R), #graspable(O1), #graspable(O2), #appliances(A).

% impossible to switch off an appliance before finding it.
-occurs(switchoff(R,A),I):- not holds(at(R,L),I), holds(on(A,L),I), #agent(R), #appliances(A), #furniture_surfaces(L).

% impossible to switch off an appliance if it is already switched_off.
-occurs(switchoff(R,A),I):- not holds(switchedon(A),I), #agent(R), #appliances(A).

%% open
% impossible to open if two objects are already in the hand of the agent
-occurs(open(R,E),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2, #agent(R), #graspable(O1), #graspable(O2), #electricals(E).

% impossible to open the door of an electrical if the agent is not in the same location as the electrical.
-occurs(open(R,E),I):- not holds(at(R,L),I), holds(on(E,L),I), #agent(R), #electricals(E), #furniture_surfaces(L).

% impossible to open a door if it is already opened.
-occurs(open(R,E),I):- holds(opened(E),I), #agent(R), #electricals(E).

% impossible to open the door of an electrical if it is not switched_off.
-occurs(open(R,E),I) :- holds(switchedon(E),I), #agent(R), #electricals(E).

%% close
% impossible to close if two objects are already in the hand of the agent
-occurs(close(R,E),I):- holds(grabbed(R,O1),I), holds(grabbed(R,O2),I), O1 != O2, #agent(R), #graspable(O1), #graspable(O2), #electricals(E).

% impossible to close the door of an electrical before finding it.
-occurs(close(R,E),I):- not holds(at(R,L),I), holds(on(E,L),I), #agent(R), #electricals(E), #furniture_surfaces(L).

% impossible to close a door if it is already closed.
-occurs(close(R,E),I):- not holds(opened(E),I), #agent(R), #electricals(E).

% ------------------------ inertial axioms ------------------------%

holds(F,I+1) :- #inertial_f(F), holds(F,I), not -holds(F,I+1).
-holds(F,I+1) :- #inertial_f(F), -holds(F,I), not holds(F,I+1).

% ------------------------------ CWA ------------------------------%

-occurs(A,I) :- not occurs(A,I).
-holds(F,0) :- #inertial_f(F), not holds(F,0).
-holds(F,I) :- #defined_f(F), not holds(F,I).

% --------------------------- planning ---------------------------%

% to achieve success the system should satisfies the goal. Failure is not acceptable
success :- goal(I), I <= n.
:- not success, current_step(I0), planning(I0).

% consider the occurrence of exogenous actions when they are absolutely necessary for resolving a conflict
occurs(A,I) :+ #agent_actions(A), #step(I), current_step(I0), planning(I0), I0 <= I.

% agent can not execute parallel actions
-occurs(A1,I) :- occurs(A2,I), A1 != A2, #agent_actions(A1), #agent_actions(A2).

% an action should occur at each time step until the goal is achieved
something_happened(I1) :- current_step(I0), planning(I0), I0 <= I1, occurs(A,I1), #agent_actions(A).
:- not something_happened(I), something_happened(I+1), I0 <= I, current_step(I0), planning(I0).

%%%--------------------------------------------------------------%%%

planning(I) :- current_step(I).

planning(0).
current_step(0).

%%%--------------------------------------------------------------%%%

next_to(counter_one, kitchentable).
next_to(counter_one, counter_three).
next_to(counter_three, kitchentable).
next_to(kitchentable,kitchen_smalltable).
next_to(kitchen_smalltable, bedroom_desk).
next_to(bedroom_desk, bedroom_coffeetable).
next_to(kitchen_smalltable, livingroom_desk).
next_to(livingroom_desk, livingroom_coffeetable).

% --------------- %

goal(I) :- holds(made(coffee),I), holds(on(coffeepot,livingroom_desk),I), holds(on(mug,livingroom_desk),I).
holds(agent_at(human,kitchen_smalltable),0).
holds(at(ahagent1,livingroom_coffeetable),0).
holds(agent_at(ahagent2,bedroom_coffeetable),0).
holds(agent_hand(human,coffeepot),0).
holds(agent_hand(ahagent2,plum),0).
holds(agent_hand(ahagent2,candybar),0).
holds(on(coffeemaker,counter_three),0).
holds(on(mug,livingroom_desk),0).
holds(on(dishwasher,counter_three),0).
-holds(opened(dishwasher),0).
-holds(switchedon(dishwasher),0).
holds(switchedon(coffeemaker),0).
holds(made(coffee),0).
occurs(exo_put(human,coffeepot,livingroom_desk),1).
display
occurs.
