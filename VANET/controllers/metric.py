from collections import defaultdict


class MetricsController:
    """Gathers and store all statistics metrics in our system."""

    def __init__(self):
        # General Metrics
        self.migrations_count = 0  # Total number of migrations happened in system.
        self.deadline_misses = 0  # Total number of deadline misses happened in system.
        self.total_tasks = 0  # Total number of tasks processed in system.
        self.cloud_tasks = 0  # Total number of tasks offloaded to cloud server.
        self.completed_tasks = 0  # Total number of tasks completed in system.

        # Per Node Metrics
        self.node_task_counts: dict[str, int] = defaultdict(int)

        # Per Step Metrics
        self.migration_counts_per_step: list[int] = []
        self.deadline_misses_per_step: list[int] = []
        self.completed_task_per_step: list[int] = []
        self.current_step_migrations = 0
        self.current_step_deadline_misses = 0
        self.current_step_completed_tasks = 0

    def inc_completed_task(self):
        self.current_step_completed_tasks += 1
        self.completed_tasks += 1

    def inc_migration(self):
        self.current_step_migrations += 1
        self.migrations_count += 1

    def inc_deadline_miss(self):
        self.current_step_deadline_misses += 1
        self.deadline_misses += 1

    def inc_total_tasks(self):
        self.total_tasks += 1

    def inc_node_tasks(self, node_id: str):
        self.node_task_counts[node_id] += 1

    def inc_cloud_tasks(self):
        self.cloud_tasks += 1

    def flush(self):
        self.migration_counts_per_step.append(self.current_step_migrations)
        self.deadline_misses_per_step.append(self.current_step_deadline_misses)
        self.completed_task_per_step.append(self.current_step_completed_tasks)
        self.current_step_deadline_misses = 0
        self.current_step_migrations = 0
        self.current_step_completed_tasks = 0

    def log_metrics(self):
        print("Metrics:")
        print(f"\tTotal migrations: {self.migrations_count}")
        print(f"\tTotal deadline misses: {self.deadline_misses}")
        print(f"\tTotal cloud tasks: {self.cloud_tasks}")
        print(f"\tTotal completed tasks: {self.completed_tasks}")
        print(f"\tTotal tasks: {self.total_tasks}")
        if self.total_tasks != 0:
            print(f"\tMigration ratio: {'{:.3f}'.format(self.migrations_count * 100 / self.total_tasks)}%")
            print(f"\tDeadline miss ratio: {'{:.3f}'.format(self.deadline_misses * 100 / self.total_tasks)}%")
