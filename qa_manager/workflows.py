from BaseAgent import *


if __name__ == "__main__":

    pool = AgentPool()
    pool.register([
        QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
        QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
        QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
        RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),
        DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
        AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
        IterativeWorkflowAgent(AGENT_CONFIG['IterativeWorkflowAgent'])
    ])

    # workflow = [
    #     # "QueryDecompositionAgentSerial",
    #     "RetrievalAgent",
    #     "AnswerGenerationAgent"
    # ]

    # query = "What nationality was James Henry Miller's wife?"

    # # workflow logs
    # logger.info(f"====== Workflow ======")
    # for agent_name in workflow:
    #     logger.info(f"\t==> {agent_name}")

    # # query logs
    # logger.info(f"\n====== Query ======")
    # logger.info(f"\t==> {query}\n")

    # logger.info(f"====== Starting Process... ======")
    # wf = AgentWorkflow(pool)
    # final_context = wf.run(query, workflow)

    # logger.info(f"\n====== Token Usage ======")
    # tracker = TokenUsageTracker()
    # logger.info(f'Token Usage: {tracker.get_usage()}')


    query_workflow_pairs = [
        {
            "query": "What nationality was James Henry Miller's wife?",
            "workflow": [
                # "QueryRewriteAgent",
                "RetrievalAgent",
                # "DocumentSelectionAgent",
                "AnswerGenerationAgent"
            ]
        },
        {
            "query": "Where did Albert Einstein die?",
            "workflow": [
                "RetrievalAgent",
                "AnswerGenerationAgent"
            ]
        },
        {
            "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "workflow": [
                "RetrievalAgent",
                "AnswerGenerationAgent"
            ]
        },
        # {
        #     "query": "What nationality was James Henry Miller's wife?",
        #     "workflow": [
        #         "QueryRewriteAgent",
        #         "RetrievalAgent",
        #         # "DocumentSelectionAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Where did Albert Einstein die?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "What nationality was James Henry Miller's wife?",
        #     "workflow": [
        #         "QueryRewriteAgent",
        #         "RetrievalAgent",
        #         # "DocumentSelectionAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Where did Albert Einstein die?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "What nationality was James Henry Miller's wife?",
        #     "workflow": [
        #         "QueryRewriteAgent",
        #         "RetrievalAgent",
        #         # "DocumentSelectionAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Where did Albert Einstein die?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
        # {
        #     "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        #     "workflow": [
        #         "RetrievalAgent",
        #         "AnswerGenerationAgent"
        #     ]
        # },
    ]

    batch_runner = BatchAgentWorkflow(pool, max_workers=4)
    final_contexts = batch_runner.run_batch(query_workflow_pairs)

    logger.info(f"\n====== Token Usage Summary ======")
    tracker = TokenUsageTracker()
    logger.info(f'Token Usage: {tracker.get_usage()}')

    # Optionally print/return final results
    for context in final_contexts:
        print(f"\nQuery: {context.get('query', 'N/A')}")
        if 'error' in context:
            print(f"Error: {context['error']}")
        else:
            print(f"Final Context: {context}")
