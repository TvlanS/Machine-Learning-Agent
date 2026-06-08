%%{init: {'theme':'base', 'flowchart': {'nodeSpacing': 45, 'rankSpacing': 70, 'curve': 'basis'}}}%%
flowchart LR
    subgraph AGENT1 [Data Transformation Agent]
        direction LR
        START([**Start**]) --> LOAD_DATA[**Load dataset**]
        LOAD_DATA --> STEP1[**Describe data**]
        STEP1 --> STEP2[**Read description**]
        STEP2 --> EXCLUDE{**Identify features to exclude**}
        EXCLUDE --> ASK_USER[**Ask user to drop**]
        ASK_USER --> USER_DEC{**Confirmed?**}
        USER_DEC -->|Yes| DROP[**Drop features**]
        USER_DEC -->|No| KEEP[**Keep features**]
        DROP --> TRANSFORM{**Need one-hot?**}
        KEEP --> TRANSFORM
        TRANSFORM -->|Yes| OHE[**Apply OHE**]
        TRANSFORM -->|No| CORREL[**Correlation**]
        OHE --> CORREL
        CORREL --> HIGH_CORR[**Find highest correlation**]
        HIGH_CORR --> DESCRIBE_NEW[**Describe transformed data**]
        DESCRIBE_NEW --> PICK_SETS[**Pick 3 feature sets**]
        PICK_SETS --> WRITE_MD[**Write summary + JSON**]
    end

    WRITE_MD --> CHECK_FAIL{**Any step failed?**}
    CHECK_FAIL -->|Yes| REPORT_MGR[**Report to manager**]
    REPORT_MGR --> END_FAIL([**Deal failed**])
    CHECK_FAIL -->|No| END_SUCCESS([**Output to Agent 2**])

    classDef dataPrep fill:#d9eaf7,stroke:#5e8ab4,stroke-width:1px,color:#1e3a5f
    classDef decision fill:#fff0d6,stroke:#c47f2e,stroke-width:1px,color:#7a4d1a
    classDef failure fill:#ffe0d9,stroke:#bf5b3a,stroke-width:1px,color:#8b3c1c
    classDef terminal fill:#ede2f7,stroke:#7b5aa6,stroke-width:1px,color:#3e2c59
    classDef handoff fill:#e8e8e8,stroke:#9e9e9e,stroke-width:1px,color:#555555

    class STEP1,STEP2,DROP,KEEP,OHE,CORREL,HIGH_CORR,DESCRIBE_NEW,PICK_SETS,WRITE_MD dataPrep
    class EXCLUDE,USER_DEC,TRANSFORM,CHECK_FAIL decision
    class REPORT_MGR failure
    class START,END_FAIL,END_SUCCESS terminal
    class LOAD_DATA handoff
    class ASK_USER decision

    style AGENT1 stroke-dasharray: 5 5, stroke:#7f8c8d, stroke-width:2px, fill:#faf9f7


    %%{init: {'theme':'base', 'flowchart': {'nodeSpacing': 45, 'rankSpacing': 70, 'curve': 'basis'}}}%%
flowchart LR
    subgraph AGENT2 [Machine Learning Agent]
        direction RL
        INPUT[**Receive from Agent 1**] --> READ_SUMMARY[**Read summary.md**]
        READ_SUMMARY --> SMALL_SET[**Pick smallest feature set**]
        SMALL_SET --> RUN_MODEL[**Run ML model**]
        RUN_MODEL --> EVAL[**Assess strengths/weaknesses**]
        EVAL --> NEXT_SET{**More sets?**}
        NEXT_SET -->|Yes| SMALL_SET
        NEXT_SET -->|No| RESULTS_OK{**Results desired?**}
        RESULTS_OK -->|No| TRY_OTHER[**Try other combos**]
        TRY_OTHER --> RUN_MODEL
        RESULTS_OK -->|Yes| CHECK_BALANCE{**Balancing needed?**}
        CHECK_BALANCE -->|Yes| STUDY_BALANCE[**Study balancing**]
        STUDY_BALANCE --> WRITE_FINAL[**Write modelling_summary.md**]
        CHECK_BALANCE -->|No| WRITE_FINAL
    end

    WRITE_FINAL --> CHECK_FAIL2{**Any step failed?**}
    CHECK_FAIL2 -->|Yes| REPORT_MGR2[**Report to manager**]
    REPORT_MGR2 --> END_FAIL([**Deal failed**])
    CHECK_FAIL2 -->|No| END_SUCCESS([**Deal closed**])

    classDef mlTask fill:#d8f0e6,stroke:#388e6f,stroke-width:1px,color:#1e4a3b
    classDef decision fill:#fff0d6,stroke:#c47f2e,stroke-width:1px,color:#7a4d1a
    classDef failure fill:#ffe0d9,stroke:#bf5b3a,stroke-width:1px,color:#8b3c1c
    classDef terminal fill:#ede2f7,stroke:#7b5aa6,stroke-width:1px,color:#3e2c59
    classDef handoff fill:#e8e8e8,stroke:#9e9e9e,stroke-width:1px,color:#555555

    class READ_SUMMARY,RUN_MODEL,EVAL,TRY_OTHER,STUDY_BALANCE,WRITE_FINAL mlTask
    class NEXT_SET,RESULTS_OK,CHECK_BALANCE,CHECK_FAIL2 decision
    class REPORT_MGR2 failure
    class END_FAIL,END_SUCCESS terminal
    class SMALL_SET mlTask
    class INPUT handoff

    style AGENT2 stroke-dasharray: 5 5, stroke:#7f8c8d, stroke-width:2px, fill:#faf9f7