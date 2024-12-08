import json
import logging
import time
import math
import sys
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


class MultiLevelDrainTrainer:
    def __init__(self, train_data_loader, test_data_loader, base_config_path, levels, trained_models_dir):
        """
        Initialize the MultiLevelDrainTrainer.

        :param data_loader: An iterable DataLoader to read log lines.
        :param base_config_path: Path to the base configuration file for Drain.
        :param levels: List of dictionaries defining hyper-parameters for each level.
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.base_config_path = base_config_path
        self.levels = levels
        self.models = []
        self.statistics = []  # Store statistics for each level
        self.trained_models_dir = trained_models_dir

    def _load_config(self, level_params):
        """
        Load the base configuration and apply level-specific hyper-parameters.

        :param level_params: Dictionary of hyper-parameters for this level.
        :return: A configured TemplateMinerConfig instance.
        """
        config = TemplateMinerConfig()
        config.load(self.base_config_path)
        for param, value in level_params.items():
            setattr(config, param, value)
        return config

    def train_level(self, level_idx, config):
        """
        Train a single level of Drain with the given configuration.

        :param level_idx: Index of the level being trained.
        :param config: Configured TemplateMinerConfig instance.
        :return: A trained TemplateMiner instance.
        """
        logger.info(f"Training Drain model for Level {level_idx} with config: {config.__dict__}")
        persistence = FilePersistence(os.path.join(self.trained_models_dir, f"state_level_{level_idx}.bin"))
        template_miner = TemplateMiner(persistence, config)

        # for file_idx, chunk in self.train_data_loader:
        #     for line in chunk:
        #         line = line.strip()
        #         template_miner.add_log_message(line)
                # masked_message = template_miner.masker.mask(line)
                # tokens = template_miner.drain.get_content_as_tokens(masked_message)
                # print(f"Original: {line}")
                # print(f"Masked: {" ".join(tokens)}")

        return template_miner

    def collect_statistics(self, level_idx, model):
        """
        Collect frequency and rarity statistics for templates after training is complete.

        :param level_idx: Index of the level being analyzed.
        :param model: Trained TemplateMiner instance.
        :return: A list of statistics for each template.
        """
        logger.info(f"Collecting statistics for Level {level_idx}...")
        stats = {}
        total_lines = 0
        template_id_template_map = {}
        total_files = set()
        # Reprocess training data to collect statistics
        for file_idx, chunk in self.test_data_loader:
            total_files.add(file_idx)
            if file_idx not in stats:
                stats[file_idx] = {}

            for line in chunk:
                line = line.strip()
                cluster = model.match(line)
                if cluster is None:
                    continue
                else:
                    cluster_id = cluster.cluster_id
                    template = cluster.get_template()
                    if cluster_id not in template_id_template_map:
                        template_id_template_map[cluster_id] = {"template": template, "count": 0, "files": set()}
                    template_id_template_map[cluster_id]["count"] += 1
                    template_id_template_map[cluster_id]["files"].add(file_idx)
                    if template not in stats[file_idx]:
                        stats[file_idx][template] = 0
                    stats[file_idx][template] += 1
                total_lines += 1

        for template_id, template_stats in template_id_template_map.items():
            try:
                template = template_stats["template"]
                template_stats["occurrence_ratio"] = len(template_stats["files"]) / len(total_files)
                template_stats["average_frequency_per_file"] = template_stats["count"] / len(total_files)

                # Compute Log-Likelihood Ratio (LLR)
                p_obs = template_stats["count"] / total_lines
                p_exp = 1 / len(template_id_template_map)
                template_stats["log_likelihood_ratio"] = template_stats["count"] * math.log(p_obs / p_exp) if p_obs > 0 else 0

                probabilities = []
                for idx in stats:
                    if stats[idx][template]:
                        probabilities.append(stats[idx][template])
                sum_probabilities = sum(probabilities)
                probabilities = [p / sum_probabilities for p in probabilities]
                entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
                template_stats["entropy"] = entropy

            except:
                continue

        self.statistics.append({"level": level_idx, "stats": template_id_template_map})
        return stats

    def train(self):
        """
        Train the Drain models for all levels and collect statistics.
        """
        for level_idx, level_params in enumerate(self.levels, start=1):
            config = self._load_config(level_params)
            model = self.train_level(level_idx, config)
            self.models.append(model)
            self.collect_statistics(level_idx, model)

    def summarize_models(self):
        """
        Summarize the clusters for all trained models.
        """
        for level_idx, model in enumerate(self.models, start=1):
            logger.info(f"--- Summary for Level {level_idx} ---")
            sorted_clusters = sorted(model.drain.clusters, key=lambda c: c.size, reverse=True)
            for cluster in sorted_clusters[:10]:  # Show top 10 clusters
                logger.info(cluster)
            print(f"Prefix Tree for Level {level_idx}:")
            model.drain.print_tree()
            model.profiler.report(0)
