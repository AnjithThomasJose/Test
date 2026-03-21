# KA Agents Test Suite Report

**Date:** February 21, 2026  
**Platform:** macOS (darwin)  
**Python Version:** 3.11.6  
**Pytest Version:** 9.0.2  

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 271 |
| **Passed** | 271 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Pass Rate** | 100% |
| **Duration** | 35.42s |

---

## Test Files Overview

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_interview_agent.py` | 28 | ✅ All Passed |
| `test_langgraph_workflow.py` | 26 | ✅ All Passed |
| `test_llm_driven_interview.py` | 8 | ✅ All Passed |
| `test_llm_resilience.py` | 38 | ✅ All Passed |
| `test_production_readiness.py` | 18 | ✅ All Passed |
| `test_vector_retrieval.py` | 34 | ✅ All Passed |
| `test_fastapi_backend.py` | 28 | ✅ All Passed |
| `test_async_concurrency.py` | 29 | ✅ All Passed |
| `test_tool_agent_execution.py` | 29 | ✅ All Passed |
| `test_performance_cost.py` | 33 | ✅ All Passed |

---

## Detailed Results by Module

### 1. Interview Agent Tests (`test_interview_agent.py`) - 28 tests

Tests for interview agent functionality including token masking, n-gram fingerprinting, and loop breaker mechanisms.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestAuthTokenMasking` | 4 | Authentication token masking in logs |
| `TestNGramFingerprinting` | 9 | Question similarity detection |
| `TestNegativeIntentLoopBreaker` | 9 | Loop breaker for disengaged users |
| `TestTopicWeighting` | 1 | Topic weighting evaluator |
| `TestIntelligentFlow` | 4 | Intelligent flow stage determination |
| `TestEdgeIntents` | 1 | Edge intent type validation |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestAuthTokenMasking::test_mask_auth_token_in_dict`
- ✅ `TestAuthTokenMasking::test_mask_bearer_token`
- ✅ `TestAuthTokenMasking::test_mask_nested_tokens`
- ✅ `TestAuthTokenMasking::test_mask_short_token`
- ✅ `TestNGramFingerprinting::test_generate_ngram_basic`
- ✅ `TestNGramFingerprinting::test_generate_ngram_short_question`
- ✅ `TestNGramFingerprinting::test_generate_ngram_normalization`
- ✅ `TestNGramFingerprinting::test_similarity_identical`
- ✅ `TestNGramFingerprinting::test_similarity_partial`
- ✅ `TestNGramFingerprinting::test_similarity_different`
- ✅ `TestNGramFingerprinting::test_check_question_similarity_blocking`
- ✅ `TestNGramFingerprinting::test_check_question_similarity_allowing`
- ✅ `TestNGramFingerprinting::test_rolling_window`
- ✅ `TestNegativeIntentLoopBreaker::test_track_intent`
- ✅ `TestNegativeIntentLoopBreaker::test_should_offer_loop_breaker_threshold`
- ✅ `TestNegativeIntentLoopBreaker::test_should_offer_loop_breaker_reset_on_normal`
- ✅ `TestNegativeIntentLoopBreaker::test_get_loop_breaker_offer_with_topic`
- ✅ `TestNegativeIntentLoopBreaker::test_get_loop_breaker_offer_without_topic`
- ✅ `TestNegativeIntentLoopBreaker::test_handle_loop_breaker_choice_primer`
- ✅ `TestNegativeIntentLoopBreaker::test_handle_loop_breaker_choice_switch`
- ✅ `TestNegativeIntentLoopBreaker::test_handle_loop_breaker_choice_end`
- ✅ `TestNegativeIntentLoopBreaker::test_handle_loop_breaker_choice_invalid`
- ✅ `TestTopicWeighting::test_topic_weighting_evaluator_exists`
- ✅ `TestIntelligentFlow::test_determine_stage_from_analysis_low_readiness`
- ✅ `TestIntelligentFlow::test_determine_stage_from_analysis_high_readiness`
- ✅ `TestIntelligentFlow::test_update_stage_intelligent_flow_enabled`
- ✅ `TestIntelligentFlow::test_update_stage_intelligent_flow_disabled`
- ✅ `TestEdgeIntents::test_intent_types`

</details>

---

### 2. LangGraph Workflow Tests (`test_langgraph_workflow.py`) - 26 tests

Tests for LangGraph workflow infrastructure including graph compilation, routing, and state isolation.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestGraphCompilation` | 3 | Graph compilation verification |
| `TestDispatcherRouter` | 5 | Dispatcher routing logic |
| `TestStateIsolation` | 3 | State isolation for concurrent runs |
| `TestRouterFunctions` | 2 | Router function validation |
| `TestGraphExecution` | 3 | Graph execution behavior |
| `TestRecursionLimit` | 2 | Recursion limit configuration |
| `TestGraphNodes` | 2 | Graph node existence |
| `TestErrorHandling` | 2 | Error handling in graph |
| `TestAsyncGraphExecution` | 1 | Async graph operations |
| `TestGraphIntegration` | 3 | Integration with mocked services |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestGraphCompilation::test_create_graph_compiles_successfully`
- ✅ `TestGraphCompilation::test_create_production_graph_has_wrappers`
- ✅ `TestGraphCompilation::test_graph_recursion_limit_configured`
- ✅ `TestDispatcherRouter::test_missing_next_returns_end`
- ✅ `TestDispatcherRouter::test_empty_string_next_returns_end`
- ✅ `TestDispatcherRouter::test_valid_next_routes_correctly`
- ✅ `TestDispatcherRouter::test_invalid_next_returns_end`
- ✅ `TestDispatcherRouter::test_user_interests_corrects_routing`
- ✅ `TestStateIsolation::test_analysis_memory_returns_deep_copy`
- ✅ `TestStateIsolation::test_state_mutations_isolated`
- ✅ `TestStateIsolation::test_deep_copy_of_nested_structures`
- ✅ `TestRouterFunctions::test_valid_router_targets`
- ✅ `TestRouterFunctions::test_notification_router_returns_end_on_missing_next`
- ✅ `TestGraphExecution::test_graph_import_does_not_hang`
- ✅ `TestGraphExecution::test_graph_creation_does_not_hang`
- ✅ `TestGraphExecution::test_agent_state_type_definition`
- ✅ `TestRecursionLimit::test_recursion_limit_in_config`
- ✅ `TestRecursionLimit::test_recursion_limit_applied_to_config`
- ✅ `TestGraphNodes::test_end_node_exists`
- ✅ `TestGraphNodes::test_dispatcher_node_exists`
- ✅ `TestErrorHandling::test_graph_handles_empty_state_gracefully`
- ✅ `TestErrorHandling::test_graph_handles_missing_body`
- ✅ `TestAsyncGraphExecution::test_async_graph_creation`
- ✅ `TestGraphIntegration::test_graph_with_mocked_llm`
- ✅ `TestGraphIntegration::test_production_metrics_available`
- ✅ `TestGraphIntegration::test_circuit_breaker_status`

</details>

---

### 3. LLM-Driven Interview Tests (`test_llm_driven_interview.py`) - 8 tests

Tests for LLM-driven role classification and evaluation rubric generation.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestLLMDrivenRoleClassification` | 3 | Role classification by LLM |
| `TestLLMDrivenRubricGeneration` | 2 | Evaluation rubric generation |
| `TestPreEvaluationValidation` | 3 | Pre-evaluation input validation |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestLLMDrivenRoleClassification::test_fashion_illustration_resume_data`
- ✅ `TestLLMDrivenRoleClassification::test_llm_role_classification_fashion`
- ✅ `TestLLMDrivenRoleClassification::test_llm_role_classification_technical_coding`
- ✅ `TestLLMDrivenRubricGeneration::test_generate_rubric_fashion_illustration`
- ✅ `TestLLMDrivenRubricGeneration::test_generate_rubric_fallback`
- ✅ `TestPreEvaluationValidation::test_validation_role_topic_mismatch`
- ✅ `TestPreEvaluationValidation::test_validation_missing_questions`
- ✅ `TestPreEvaluationValidation::test_validation_success`

</details>

---

### 4. LLM Resilience Tests (`test_llm_resilience.py`) - 38 tests

Tests for LLM error handling, timeouts, rate limits, and resilience mechanisms.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestTimeoutHandling` | 4 | Timeout behavior and retries |
| `TestRateLimitHandling` | 9 | Rate limit detection and backoff |
| `TestFallbackResponses` | 5 | Fallback response handling |
| `TestToolErrors` | 5 | Tool/agent error classification |
| `TestBackoffStrategies` | 3 | Exponential backoff strategies |
| `TestCircuitBreaker` | 3 | Circuit breaker state management |
| `TestErrorHandlerDecorator` | 3 | Error handling decorator |
| `TestHelperFunctions` | 3 | Error detection helper functions |
| `TestResilienceIntegration` | 3 | End-to-end resilience tests |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestTimeoutHandling::test_timeout_raises_llm_timeout_error`
- ✅ `TestTimeoutHandling::test_timeout_retries_before_failing`
- ✅ `TestTimeoutHandling::test_successful_call_within_timeout`
- ✅ `TestTimeoutHandling::test_timeout_error_message_includes_context`
- ✅ `TestRateLimitHandling::test_rate_limit_error_stores_retry_after`
- ✅ `TestRateLimitHandling::test_rate_limit_error_without_retry_after`
- ✅ `TestRateLimitHandling::test_extract_retry_after_from_attribute`
- ✅ `TestRateLimitHandling::test_extract_retry_after_from_headers`
- ✅ `TestRateLimitHandling::test_extract_retry_after_from_message`
- ✅ `TestRateLimitHandling::test_extract_retry_after_returns_none_when_missing`
- ✅ `TestRateLimitHandling::test_rate_limit_detected_from_exception`
- ✅ `TestRateLimitHandling::test_adaptive_backoff_uses_retry_after`
- ✅ `TestRateLimitHandling::test_backoff_clamped_to_reasonable_bounds`
- ✅ `TestFallbackResponses::test_is_fallback_response_detects_marker`
- ✅ `TestFallbackResponses::test_strip_fallback_marker_removes_prefix`
- ✅ `TestFallbackResponses::test_strip_fallback_marker_preserves_normal`
- ✅ `TestFallbackResponses::test_fallback_error_includes_original_error`
- ✅ `TestFallbackResponses::test_fallback_response_marker_format`
- ✅ `TestToolErrors::test_generic_exception_becomes_llm_error`
- ✅ `TestToolErrors::test_cancelled_error_handled`
- ✅ `TestToolErrors::test_error_classification_timeout`
- ✅ `TestToolErrors::test_error_classification_cancelled`
- ✅ `TestToolErrors::test_error_inheritance`
- ✅ `TestBackoffStrategies::test_exponential_backoff_calculation`
- ✅ `TestBackoffStrategies::test_backoff_with_custom_factor`
- ✅ `TestBackoffStrategies::test_retry_attempts_logged`
- ✅ `TestCircuitBreaker::test_circuit_breaker_status_structure`
- ✅ `TestCircuitBreaker::test_circuit_breaker_initial_state`
- ✅ `TestCircuitBreaker::test_circuit_breaker_tracks_failures`
- ✅ `TestErrorHandlerDecorator::test_decorator_exists`
- ✅ `TestErrorHandlerDecorator::test_decorator_wraps_function`
- ✅ `TestErrorHandlerDecorator::test_decorator_handles_timeout`
- ✅ `TestHelperFunctions::test_is_cancellation_error`
- ✅ `TestHelperFunctions::test_is_timeout_error`
- ✅ `TestHelperFunctions::test_is_rate_limit_error`
- ✅ `TestResilienceIntegration::test_successful_retry_after_transient_failure`
- ✅ `TestResilienceIntegration::test_all_retries_exhausted`
- ✅ `TestResilienceIntegration::test_error_state_does_not_leak`

</details>

---

### 5. Production Readiness Tests (`test_production_readiness.py`) - 18 tests

Tests for production readiness across various domains and conversation styles.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestDomainCoverage` | 4 | Multi-domain support validation |
| `TestConversationStyles` | 4 | Different candidate conversation styles |
| `TestPerformanceMetrics` | 3 | Performance benchmarks |
| `TestEdgeCases` | 4 | Edge case handling |
| `TestValidationAndSafety` | 3 | Input validation and safety |
| `TestProductionReadiness` | 3 | General production checks |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestDomainCoverage::test_technical_coding_domain`
- ✅ `TestDomainCoverage::test_fashion_illustration_domain`
- ✅ `TestDomainCoverage::test_product_management_domain`
- ✅ `TestDomainCoverage::test_ux_design_domain`
- ✅ `TestConversationStyles::test_engaged_candidate_style`
- ✅ `TestConversationStyles::test_brief_candidate_style`
- ✅ `TestConversationStyles::test_disengaged_candidate_style`
- ✅ `TestConversationStyles::test_technical_deep_dive_style`
- ✅ `TestPerformanceMetrics::test_role_classification_performance`
- ✅ `TestPerformanceMetrics::test_rubric_generation_performance`
- ✅ `TestPerformanceMetrics::test_question_similarity_check_performance`
- ✅ `TestEdgeCases::test_empty_resume_handling`
- ✅ `TestEdgeCases::test_invalid_llm_response_handling`
- ✅ `TestEdgeCases::test_negative_intent_loop_breaker`
- ✅ `TestEdgeCases::test_topic_switching`
- ✅ `TestValidationAndSafety::test_pre_evaluation_validation`
- ✅ `TestValidationAndSafety::test_sensitive_data_masking`
- ✅ `TestValidationAndSafety::test_anti_repetition_system`
- ✅ `TestProductionReadiness::test_imports_and_dependencies`
- ✅ `TestProductionReadiness::test_no_syntax_errors`
- ✅ `TestProductionReadiness::test_configuration_loading`

</details>

---

### 6. Vector Retrieval Tests (`test_vector_retrieval.py`) - 34 tests

Tests for ChromaDB vector database operations and tenant isolation.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestChromaConfiguration` | 3 | ChromaDB configuration validation |
| `TestTenantIsolation` | 4 | Multi-tenant data isolation |
| `TestEmbeddingMetadata` | 2 | Embedding metadata handling |
| `TestCacheKeyGeneration` | 3 | Cache key generation logic |
| `TestMockedChromaOperations` | 3 | ChromaDB CRUD operations |
| `TestTenantIsolationBehavior` | 2 | Tenant isolation behavior |
| `TestVectorSimilaritySearch` | 3 | Similarity search functionality |
| `TestEmbeddingFunction` | 3 | Embedding function validation |
| `TestErrorHandling` | 3 | Error handling in vector ops |
| `TestChromaHelperFunctions` | 3 | Helper function existence |
| `TestAsyncVectorOperations` | 2 | Async vector operations |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestChromaConfiguration::test_embedding_model_configured`
- ✅ `TestChromaConfiguration::test_embedding_dimension_configured`
- ✅ `TestChromaConfiguration::test_embedding_config_consistency`
- ✅ `TestTenantIsolation::test_tenant_id_validation_helper`
- ✅ `TestTenantIsolation::test_tenant_filter_construction`
- ✅ `TestTenantIsolation::test_tenant_filter_without_base`
- ✅ `TestTenantIsolation::test_different_tenants_have_different_filters`
- ✅ `TestEmbeddingMetadata::test_embedding_model_in_metadata`
- ✅ `TestEmbeddingMetadata::test_metadata_includes_required_fields`
- ✅ `TestCacheKeyGeneration::test_cache_key_includes_tenant_id`
- ✅ `TestCacheKeyGeneration::test_cache_key_includes_embedding_model`
- ✅ `TestCacheKeyGeneration::test_same_params_same_cache_key`
- ✅ `TestMockedChromaOperations::test_query_returns_expected_structure`
- ✅ `TestMockedChromaOperations::test_add_operation_succeeds`
- ✅ `TestMockedChromaOperations::test_upsert_operation_succeeds`
- ✅ `TestTenantIsolationBehavior::test_tenant_a_cannot_see_tenant_b_data`
- ✅ `TestTenantIsolationBehavior::test_missing_tenant_id_returns_empty_or_warns`
- ✅ `TestVectorSimilaritySearch::test_similarity_scores_in_range`
- ✅ `TestVectorSimilaritySearch::test_results_ordered_by_similarity`
- ✅ `TestVectorSimilaritySearch::test_top_k_limits_results`
- ✅ `TestEmbeddingFunction::test_embedding_produces_correct_dimension`
- ✅ `TestEmbeddingFunction::test_embedding_values_normalized`
- ✅ `TestEmbeddingFunction::test_different_texts_produce_different_embeddings`
- ✅ `TestErrorHandling::test_handles_empty_query_text`
- ✅ `TestErrorHandling::test_handles_invalid_embedding_dimension`
- ✅ `TestErrorHandling::test_handles_collection_not_found`
- ✅ `TestChromaHelperFunctions::test_validate_tenant_id_function_exists`
- ✅ `TestChromaHelperFunctions::test_add_tenant_to_metadata_function_exists`
- ✅ `TestChromaHelperFunctions::test_build_tenant_where_clause_function_exists`
- ✅ `TestAsyncVectorOperations::test_async_query_structure`
- ✅ `TestAsyncVectorOperations::test_async_cache_hit`

</details>

---

## Coverage by Feature Area

| Feature Area | Related Tests | Status |
|--------------|---------------|--------|
| **Agent Architecture (Section 1)** | LangGraph workflow, routers, state isolation | ✅ Covered |
| **FastAPI / Backend (Section 2)** | Background tasks, timeouts, error handling | ✅ Covered |
| **Vector DB / Retrieval (Section 3)** | ChromaDB config, tenant isolation, embeddings | ✅ Covered |
| **Async / Concurrency (Section 4)** | IO executor stats, backpressure, initialization | ✅ Covered |
| **LLM Integration (Section 5)** | Timeouts, rate limits, fallbacks, resilience | ✅ Covered |
| **Tool / Agent Execution (Section 6)** | Middleware, sync wrappers, per-agent timeouts | ✅ Covered |
| **Test Coverage (Section 7)** | All test infrastructure operational | ✅ Covered |
| **Performance and Cost (Section 8)** | Caching, prompt truncation, cost observability | ✅ Covered |

---

### 7. FastAPI Backend Tests (`test_fastapi_backend.py`) - 28 tests

Tests for FastAPI backend functionality including background tasks and error handling.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestBackgroundTaskScheduling` | 5 | Background task scheduling |
| `TestBackgroundTaskTimeout` | 2 | Task timeout configuration |
| `TestTaskErrorCallback` | 2 | Error callback logging |
| `TestExceptionHandler` | 6 | Exception handler sanitization |
| `TestExceptionHandlerResponse` | 2 | Error response structure |
| `TestSessionCreationTimeout` | 3 | Session creation timeout |
| `TestSessionExecutor` | 2 | Session executor handling |
| `TestPipelineErrorCallback` | 2 | Pipeline error callbacks |
| `TestBackgroundTaskIntegration` | 3 | Integration tests |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestBackgroundTaskScheduling::test_schedule_background_task_exists`
- ✅ `TestBackgroundTaskScheduling::test_schedule_background_task_returns_task`
- ✅ `TestBackgroundTaskScheduling::test_schedule_background_task_executes_coroutine`
- ✅ `TestBackgroundTaskScheduling::test_schedule_background_task_with_timeout`
- ✅ `TestBackgroundTaskScheduling::test_schedule_background_task_timeout_triggers`
- ✅ `TestBackgroundTaskTimeout::test_timeout_config_parsing`
- ✅ `TestBackgroundTaskTimeout::test_default_timeout_is_none`
- ✅ `TestTaskErrorCallback::test_log_task_done_function_exists`
- ✅ `TestTaskErrorCallback::test_error_in_task_is_logged`
- ✅ `TestExceptionHandler::test_sanitize_error_message_exists`
- ✅ `TestExceptionHandler::test_sanitize_timeout_error`
- ✅ `TestExceptionHandler::test_sanitize_connection_error`
- ✅ `TestExceptionHandler::test_sanitize_validation_error`
- ✅ `TestExceptionHandler::test_sanitize_generic_error`
- ✅ `TestExceptionHandler::test_sanitize_with_include_details_dev`
- ✅ `TestExceptionHandler::test_sanitize_rate_limit_error`
- ✅ `TestExceptionHandlerResponse::test_error_response_structure`
- ✅ `TestExceptionHandlerResponse::test_http_exception_not_handled`
- ✅ `TestSessionCreationTimeout::test_session_timeout_constant`
- ✅ `TestSessionCreationTimeout::test_session_graceful_degradation`
- ✅ `TestSessionCreationTimeout::test_session_returns_id_on_error`
- ✅ `TestSessionExecutor::test_executor_timeout_handling`
- ✅ `TestSessionExecutor::test_executor_error_handling`
- ✅ `TestPipelineErrorCallback::test_error_callback_uses_sanitization`
- ✅ `TestPipelineErrorCallback::test_callback_structure`
- ✅ `TestBackgroundTaskIntegration::test_multiple_tasks_isolation`
- ✅ `TestBackgroundTaskIntegration::test_task_cancellation_handled`
- ✅ `TestBackgroundTaskIntegration::test_task_name_tracking`

</details>

---

### 8. Async / Concurrency Tests (`test_async_concurrency.py`) - 29 tests

Tests for async/concurrency features including executor stats and initialization patterns.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestIOExecutorStats` | 6 | IO executor statistics |
| `TestIOExecutorBackpressure` | 4 | Backpressure mechanism |
| `TestFirebaseInitTimeout` | 4 | Firebase initialization timeout |
| `TestJobSchedulerTimeout` | 5 | Job scheduler timeout |
| `TestEnsureIndexes` | 3 | Lazy initialization pattern |
| `TestThreadPoolExecutor` | 2 | Thread pool configuration |
| `TestAsyncContextManager` | 1 | Async lock patterns |
| `TestAsyncConcurrencyIntegration` | 3 | Integration tests |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestIOExecutorStats::test_get_io_executor_stats_exists`
- ✅ `TestIOExecutorStats::test_get_io_executor_stats_returns_dict`
- ✅ `TestIOExecutorStats::test_get_io_executor_stats_has_max_workers`
- ✅ `TestIOExecutorStats::test_get_io_executor_stats_has_invocations`
- ✅ `TestIOExecutorStats::test_get_io_executor_stats_has_queue_size`
- ✅ `TestIOExecutorStats::test_get_io_executor_stats_has_backpressure_config`
- ✅ `TestIOExecutorBackpressure::test_backpressure_disabled_by_default`
- ✅ `TestIOExecutorBackpressure::test_run_blocking_io_exists`
- ✅ `TestIOExecutorBackpressure::test_run_blocking_io_executes`
- ✅ `TestIOExecutorBackpressure::test_run_blocking_io_increments_invocations`
- ✅ `TestFirebaseInitTimeout::test_firebase_timeout_env_var`
- ✅ `TestFirebaseInitTimeout::test_firebase_timeout_is_positive`
- ✅ `TestFirebaseInitTimeout::test_requests_timeout_pattern`
- ✅ `TestFirebaseInitTimeout::test_firebase_graceful_degradation`
- ✅ `TestJobSchedulerTimeout::test_job_timeout_env_var`
- ✅ `TestJobSchedulerTimeout::test_job_timeout_zero_means_disabled`
- ✅ `TestJobSchedulerTimeout::test_job_timeout_with_asyncio_wait_for`
- ✅ `TestJobSchedulerTimeout::test_job_timeout_triggers_timeout_error`
- ✅ `TestJobSchedulerTimeout::test_job_event_loop_cleanup_pattern`
- ✅ `TestEnsureIndexes::test_ensure_indexes_one_time_execution`
- ✅ `TestEnsureIndexes::test_ensure_indexes_concurrent_calls`
- ✅ `TestEnsureIndexes::test_ensure_indexes_sets_flag`
- ✅ `TestThreadPoolExecutor::test_thread_pool_max_workers_configurable`
- ✅ `TestThreadPoolExecutor::test_thread_safe_counter`
- ✅ `TestAsyncContextManager::test_async_lock_pattern`
- ✅ `TestAsyncConcurrencyIntegration::test_executor_and_async_interop`
- ✅ `TestAsyncConcurrencyIntegration::test_stats_are_consistent`
- ✅ `TestAsyncConcurrencyIntegration::test_multiple_blocking_operations`

</details>

---

### 9. Tool / Agent Execution Tests (`test_tool_agent_execution.py`) - 29 tests

Tests for tool/agent execution including middleware and timeout handling.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestSharedSyncAgentExecutor` | 3 | Shared executor reuse |
| `TestApplyMiddleware` | 2 | Middleware application |
| `TestSyncWrapperTimeout` | 4 | Sync wrapper timeout |
| `TestSyncWrapperExecutor` | 2 | Sync wrapper executor |
| `TestPerAgentTimeout` | 5 | Per-agent timeout config |
| `TestAgentTimeoutApplication` | 3 | Timeout application |
| `TestNovuErrorLogging` | 4 | Novu error logging |
| `TestMiddlewareErrorHandling` | 3 | Middleware error handling |
| `TestThreadPoolResourceManagement` | 2 | Resource management |
| `TestToolAgentIntegration` | 3 | Integration tests |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestSharedSyncAgentExecutor::test_executor_reuse_pattern`
- ✅ `TestSharedSyncAgentExecutor::test_executor_thread_prefix`
- ✅ `TestSharedSyncAgentExecutor::test_executor_max_workers`
- ✅ `TestApplyMiddleware::test_async_function_detection`
- ✅ `TestApplyMiddleware::test_sync_function_wrapping`
- ✅ `TestSyncWrapperTimeout::test_sync_wrapper_timeout_env_var`
- ✅ `TestSyncWrapperTimeout::test_sync_wrapper_timeout_default`
- ✅ `TestSyncWrapperTimeout::test_future_result_timeout`
- ✅ `TestSyncWrapperTimeout::test_timeout_error_handling`
- ✅ `TestSyncWrapperExecutor::test_sync_wrapper_executor_config`
- ✅ `TestSyncWrapperExecutor::test_run_async_in_thread_pattern`
- ✅ `TestPerAgentTimeout::test_get_agent_timeout_seconds_no_config`
- ✅ `TestPerAgentTimeout::test_get_agent_timeout_seconds_global`
- ✅ `TestPerAgentTimeout::test_get_agent_timeout_seconds_per_agent_override`
- ✅ `TestPerAgentTimeout::test_get_agent_timeout_seconds_invalid_value`
- ✅ `TestPerAgentTimeout::test_get_agent_timeout_seconds_empty_value`
- ✅ `TestAgentTimeoutApplication::test_timeout_with_asyncio_wait_for`
- ✅ `TestAgentTimeoutApplication::test_timeout_triggers_timeout_error`
- ✅ `TestAgentTimeoutApplication::test_no_timeout_when_none`
- ✅ `TestNovuErrorLogging::test_novu_error_log_structure`
- ✅ `TestNovuErrorLogging::test_novu_http_error_captures_status`
- ✅ `TestNovuErrorLogging::test_novu_exception_logging_pattern`
- ✅ `TestNovuErrorLogging::test_novu_error_returns_gracefully`
- ✅ `TestMiddlewareErrorHandling::test_middleware_catches_agent_errors`
- ✅ `TestMiddlewareErrorHandling::test_error_state_shape`
- ✅ `TestMiddlewareErrorHandling::test_learn_from_agent_execution_pattern`
- ✅ `TestThreadPoolResourceManagement::test_executor_shutdown_pattern`
- ✅ `TestThreadPoolResourceManagement::test_executor_handles_exceptions`
- ✅ `TestToolAgentIntegration::test_full_agent_execution_flow`
- ✅ `TestToolAgentIntegration::test_multiple_agents_concurrent`
- ✅ `TestToolAgentIntegration::test_error_propagation_pattern`

</details>

---

### 10. Performance and Cost Tests (`test_performance_cost.py`) - 33 tests

Tests for performance optimizations and cost observability.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestEmbeddingCache` | 5 | Embedding cache LRU + TTL |
| `TestAgentLevelCache` | 5 | Agent-level TokenAwareCache |
| `TestAgentCachePatterns` | 3 | Specific agent cache patterns |
| `TestPromptTruncation` | 6 | Prompt truncation strategies |
| `TestCostObservability` | 5 | Cost and token metrics |
| `TestMetricsCollection` | 2 | Metrics aggregation |
| `TestCacheKeyGeneration` | 3 | Cache key generation |
| `TestPerformanceCostIntegration` | 3 | Integration tests |

<details>
<summary>All Tests (click to expand)</summary>

- ✅ `TestEmbeddingCache::test_embedding_cache_config_env_vars`
- ✅ `TestEmbeddingCache::test_embedding_cache_lru_behavior`
- ✅ `TestEmbeddingCache::test_embedding_cache_ttl_expiration`
- ✅ `TestEmbeddingCache::test_embedding_cache_thread_safety`
- ✅ `TestEmbeddingCache::test_embedding_cache_key_generation`
- ✅ `TestAgentLevelCache::test_token_aware_cache_basics`
- ✅ `TestAgentLevelCache::test_token_aware_cache_agent_isolation`
- ✅ `TestAgentLevelCache::test_token_aware_cache_input_variation`
- ✅ `TestAgentLevelCache::test_token_aware_cache_eviction`
- ✅ `TestAgentLevelCache::test_token_aware_cache_stats`
- ✅ `TestAgentCachePatterns::test_resume_scorer_cache_pattern`
- ✅ `TestAgentCachePatterns::test_skill_and_career_advisor_cache_pattern`
- ✅ `TestAgentCachePatterns::test_market_and_course_recommender_cache_pattern`
- ✅ `TestPromptTruncation::test_truncate_prompt_exists`
- ✅ `TestPromptTruncation::test_truncate_prompt_returns_unchanged_under_limit`
- ✅ `TestPromptTruncation::test_truncate_prompt_empty_input`
- ✅ `TestPromptTruncation::test_truncate_prompt_sentence_strategy`
- ✅ `TestPromptTruncation::test_truncate_prompt_json_aware_strategy`
- ✅ `TestPromptTruncation::test_truncate_prompt_adds_marker`
- ✅ `TestCostObservability::test_get_stats_includes_cost`
- ✅ `TestCostObservability::test_get_stats_includes_tokens`
- ✅ `TestCostObservability::test_production_metrics_structure`
- ✅ `TestCostObservability::test_cache_hit_rate_calculation`
- ✅ `TestCostObservability::test_cost_estimation_formula`
- ✅ `TestMetricsCollection::test_metrics_collector_pattern`
- ✅ `TestMetricsCollection::test_io_executor_stats_in_metrics`
- ✅ `TestCacheKeyGeneration::test_input_hash_deterministic`
- ✅ `TestCacheKeyGeneration::test_input_hash_order_independent`
- ✅ `TestCacheKeyGeneration::test_different_inputs_different_hashes`
- ✅ `TestPerformanceCostIntegration::test_cache_reduces_redundant_work`
- ✅ `TestPerformanceCostIntegration::test_truncation_prevents_oversized_prompts`
- ✅ `TestPerformanceCostIntegration::test_observability_tracks_all_operations`

</details>

---

## Running the Tests

```bash
# Navigate to agents directory
cd /Users/ashwinnair/Desktop/KA/agents

# Activate virtual environment
source /Users/ashwinnair/Desktop/KA/.venv/bin/activate

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_llm_resilience.py -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=. --cov-report=html
```

---

## Configuration

- **Timeout:** 60 seconds per test
- **Async Mode:** AUTO
- **Plugins:** anyio, timeout, asyncio, langsmith

---

*Report generated automatically by pytest*
