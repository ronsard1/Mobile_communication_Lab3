# Mobile Communications – Lab 3 (Unit 3 Mini Projects)

University of Rwanda, College of Science and Technology, School of ICT, Department of Computer Engineering
Group 3: Ronsard RUGIRANGOGA (223026673), Ivan MULISA (223026695), Dieudonne BYIRINGIRO (223026513)

The written report summarises the results; this repository holds the complete code, the printed
results, the generated signal series, the complete transition logs and all charts.

## Structure

```
Mini_Project_3_1_Reuse_Handover/
    hexgrid.py            axial hex layout, cluster labelling from (i, j), measured reuse distance
    handover.py           synthetic RSRP series and the shared handover engine (hysteresis, TTT, offsets)
    mini_project_3_1.py   part (a), part (b), lab tasks 1-6 -> results_3_1.txt, figures/
Mini_Project_3_2_Capacity_Planning/
    capacity_planner.py   CONFIG block, capacity / Q / SIR / Erlang B per N, cell splitting, sectoring
    results_3_2.txt, results_3_2.csv, figures/
Mini_Project_3_3_Handover_FSM/
    fsm.py                IDLE, RA_MSG1-4 (RACH), CONNECTED, HANDOVER; RLF and HO-failure paths;
                          naive_policy and make_hyst_ttt_policy (interchangeable)
    mini_project_3_3.py   scripted A -> B -> C drive, three policies, RACH contention check
    state_diagram.mmd     Mermaid source of the state diagram (figures/fig_state_diagram.png)
    scenario_signals.csv  the generated RSRP series of cells A, B, C
    logs/                 complete transition logs per policy (CSV)
    results_3_3.txt, figures/
```

## Run

Python 3.9 or later.

```
pip install -r requirements.txt
python Mini_Project_3_1_Reuse_Handover/mini_project_3_1.py        (~10 s)
python Mini_Project_3_2_Capacity_Planning/capacity_planner.py     (~1 s)
python Mini_Project_3_3_Handover_FSM/mini_project_3_3.py          (~3 s)
mmdc -i Mini_Project_3_3_Handover_FSM/state_diagram.mmd -o Mini_Project_3_3_Handover_FSM/figures/fig_state_diagram.png   (optional, Mermaid CLI)
```

Seeds are fixed, so the numbers match the report.
