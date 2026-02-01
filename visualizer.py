"""
Visualization module for RL training progress
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


def get_predictions(agents, splits, ml_only=False):
    """Get predictions from agents for visualization."""
    
    predictions = {}
    test_datasets = [
        (splits[1][2], splits[1][5]),
        (splits[2][2], splits[2][5]),
        (splits[3][2], splits[3][5]),
        (splits[4][2], splits[4][5]),
        (splits[5][2], splits[5][5])
    ]
    
    for idx, (agent, (X_test, y_test)) in enumerate(zip(agents, test_datasets)):
        if ml_only:
            # ML predictions only
            y_pred = agent.detector.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            # Hybrid ML+RL predictions
            y_pred_classes = []
            for sample_idx in range(len(X_test)):
                if len(X_test[sample_idx].shape) == 3:
                    sample = X_test[sample_idx].reshape(1, 28, 28, 1)
                else:
                    sample = X_test[sample_idx].reshape(1, -1)
                
                pred = agent.detector.predict(sample, verbose=0)[0]
                confidence = np.max(pred)
                pred_class = np.argmax(pred)
                
                state = np.array([
                    confidence,
                    pred_class / agent.rl_agent.action_size,
                    1 if pred_class == y_test[sample_idx] else 0,
                    np.mean(pred),
                    np.std(pred),
                    np.min(pred),
                    np.max(pred),
                    len(pred),
                    y_test[sample_idx] / agent.rl_agent.action_size,
                    sample_idx / len(X_test)
                ])
                
                rl_action = agent.rl_agent.act(state, explore=False)
                y_pred_classes.append(rl_action)
            
            y_pred_classes = np.array(y_pred_classes)
        
        predictions[idx] = (y_pred_classes, y_test)
    
    return predictions


def plot_rl_training_progress(all_agents, n_episodes, max_steps_per_episode):
    """Generate comprehensive RL training visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')

    colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'purple']
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']

    # Plot 1: Episode Rewards
    for i, agent in enumerate(all_agents):
        axes[0, 0].plot(agent.rl_agent.episode_rewards, label=agents_short[i], 
                       color=colors[i], alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Epsilon Decay
    epsilon_history = [1.0]
    eps = 1.0
    for _ in range(n_episodes):
        eps = max(0.01, eps * 0.995)
        epsilon_history.append(eps)
    axes[0, 1].plot(epsilon_history, color='darkred', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Exploration Rate (ε) Decay')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss Evolution
    for i, agent in enumerate(all_agents):
        if agent.rl_agent.losses:
            # Moving average
            window = 50
            if len(agent.rl_agent.losses) >= window:
                moving_avg = np.convolve(agent.rl_agent.losses, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(moving_avg, label=agents_short[i], color=colors[i], alpha=0.7)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Evolution (50-step MA)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Epsilon Decay (detailed)
    axes[1, 1].plot(epsilon_history, color='darkred', linewidth=2, label='Epsilon')
    axes[1, 1].axhline(y=0.01, color='orange', linestyle='--', label='Min (0.01)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon Value')
    axes[1, 1].set_title('Detailed Epsilon Decay')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rl_training_progress.png', dpi=150, bbox_inches='tight')
    print("\n✓ RL training visualization saved: rl_training_progress.png")
    plt.show()


def plot_ml_vs_hybrid_comparison(ml_results, hybrid_results, all_agents):
    """Compare ML-only vs Hybrid ML+RL performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('ML Detector vs Hybrid ML+RL System Comparison', fontsize=16, fontweight='bold')

    colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'purple']
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
    
    agent_names_list = list(ml_results.keys())

    # Plot 1: Accuracy Comparison
    ml_accs = [ml_results[name]['accuracy'] for name in agent_names_list]
    hybrid_accs = [hybrid_results[name]['accuracy'] for name in agent_names_list]
    
    x = np.arange(len(agents_short))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, ml_accs, width, label='ML Only', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, hybrid_accs, width, label='Hybrid (ML+RL)', color='coral', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy: ML vs Hybrid')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(agents_short)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (ml, hybrid) in enumerate(zip(ml_accs, hybrid_accs)):
        axes[0, 0].text(i - width/2, ml + 0.02, f'{ml:.3f}', ha='center', fontsize=9)
        axes[0, 0].text(i + width/2, hybrid + 0.02, f'{hybrid:.3f}', ha='center', fontsize=9)

    # Plot 2: Precision Comparison
    ml_prec = [ml_results[name]['precision'] for name in agent_names_list]
    hybrid_prec = [hybrid_results[name]['precision'] for name in agent_names_list]
    
    axes[0, 1].bar(x - width/2, ml_prec, width, label='ML Only', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, hybrid_prec, width, label='Hybrid (ML+RL)', color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision: ML vs Hybrid')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agents_short)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Recall Comparison
    ml_rec = [ml_results[name]['recall'] for name in agent_names_list]
    hybrid_rec = [hybrid_results[name]['recall'] for name in agent_names_list]
    
    axes[1, 0].bar(x - width/2, ml_rec, width, label='ML Only', color='steelblue', alpha=0.8)
    axes[1, 0].bar(x + width/2, hybrid_rec, width, label='Hybrid (ML+RL)', color='coral', alpha=0.8)
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall: ML vs Hybrid')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(agents_short)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: F1-Score Comparison
    ml_f1 = [ml_results[name]['f1'] for name in agent_names_list]
    hybrid_f1 = [hybrid_results[name]['f1'] for name in agent_names_list]
    
    axes[1, 1].bar(x - width/2, ml_f1, width, label='ML Only', color='steelblue', alpha=0.8)
    axes[1, 1].bar(x + width/2, hybrid_f1, width, label='Hybrid (ML+RL)', color='coral', alpha=0.8)
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score: ML vs Hybrid')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(agents_short)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ml_vs_hybrid_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison visualization saved: ml_vs_hybrid_comparison.png")
    plt.show()


def plot_improvement_summary(ml_results, hybrid_results, summary_stats):
    """Visualize the improvement brought by RL agents."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact of RL Training on System Performance', fontsize=14, fontweight='bold')

    agent_names_list = list(ml_results.keys())
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']

    # Calculate improvements
    improvements = []
    for name in agent_names_list:
        improvement = hybrid_results[name]['accuracy'] - ml_results[name]['accuracy']
        improvements.append(improvement)

    # Plot 1: Improvement per agent
    colors_improvement = ['green' if x >= 0 else 'red' for x in improvements]
    axes[0].bar(agents_short, improvements, color=colors_improvement, alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Accuracy Improvement')
    axes[0].set_title('Accuracy Gain from RL Training (Per Agent)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(improvements):
        axes[0].text(i, v + 0.01 if v >= 0 else v - 0.01, f'{v:+.4f}', 
                    ha='center', fontweight='bold')

    # Plot 2: System-level improvement
    ml_avg = summary_stats['ml_avg_accuracy']
    hybrid_avg = summary_stats['hybrid_avg_accuracy']
    
    systems = ['ML Detector\nOnly', 'Hybrid\nML+RL']
    accuracies = [ml_avg, hybrid_avg]
    colors_sys = ['steelblue', 'coral']
    
    bars = axes[1].bar(systems, accuracies, color=colors_sys, alpha=0.8, width=0.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('System-Level Performance Improvement')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.4f}\n({acc*100:.2f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement_pct = (summary_stats['improvement'] / ml_avg) * 100 if ml_avg > 0 else 0
    axes[1].text(0.5, 0.5, f'Improvement:\n+{summary_stats["improvement"]:.4f}\n({improvement_pct:.2f}%)',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('rl_improvement_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Improvement summary saved: rl_improvement_summary.png")
    plt.show()


def plot_per_agent_performance(all_agents, ml_results, hybrid_results):
    """Create detailed visualizations for each agent."""
    
    print("\n" + "="*80)
    print("GENERATING PER-AGENT VISUALIZATIONS")
    print("="*80)
    
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
    agent_names_list = list(ml_results.keys())
    
    for agent_idx, (agent, agent_name) in enumerate(zip(all_agents, agent_names_list)):
        print(f"\n📊 Generating visualizations for {agent_name}...")
        
        ml_metrics = ml_results[agent_name]
        hybrid_metrics = hybrid_results[agent_name]
        
        # Create a figure with 2x2 subplots per agent
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{agent_name} - ML vs Hybrid Performance', fontsize=14, fontweight='bold')
        
        # Plot 1: Metrics comparison (Accuracy, Precision, Recall, F1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        ml_values = [ml_metrics['accuracy'], ml_metrics['precision'], 
                     ml_metrics['recall'], ml_metrics['f1']]
        hybrid_values = [hybrid_metrics['accuracy'], hybrid_metrics['precision'], 
                        hybrid_metrics['recall'], hybrid_metrics['f1']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ml_values, width, label='ML Only', color='steelblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, hybrid_values, width, label='Hybrid (ML+RL)', color='coral', alpha=0.8)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=15)
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.0])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (ml, hybrid) in enumerate(zip(ml_values, hybrid_values)):
            axes[0, 0].text(i - width/2, ml + 0.02, f'{ml:.3f}', ha='center', fontsize=8)
            axes[0, 0].text(i + width/2, hybrid + 0.02, f'{hybrid:.3f}', ha='center', fontsize=8)
        
        # Plot 2: Improvement per metric
        improvements = [hybrid_values[i] - ml_values[i] for i in range(len(metrics))]
        colors_imp = ['green' if x >= 0 else 'red' for x in improvements]
        
        axes[0, 1].bar(metrics, improvements, color=colors_imp, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].set_ylabel('Improvement')
        axes[0, 1].set_title('RL Impact per Metric')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(improvements):
            axes[0, 1].text(i, v + 0.01 if v >= 0 else v - 0.01, f'{v:+.3f}', 
                           ha='center', fontweight='bold', fontsize=9)
        
        # Plot 3: RL Agent training stats
        if agent.rl_agent.episode_rewards:
            episodes = np.arange(len(agent.rl_agent.episode_rewards))
            axes[1, 0].plot(episodes, agent.rl_agent.episode_rewards, color='purple', linewidth=2, alpha=0.7)
            axes[1, 0].fill_between(episodes, agent.rl_agent.episode_rewards, alpha=0.3, color='purple')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Total Reward')
            axes[1, 0].set_title('RL Agent Episode Rewards')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: RL loss evolution
        if agent.rl_agent.losses:
            window = min(50, len(agent.rl_agent.losses) // 10)
            if window > 1:
                moving_avg = np.convolve(agent.rl_agent.losses, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(moving_avg, color='darkred', linewidth=2, alpha=0.7)
                axes[1, 1].fill_between(np.arange(len(moving_avg)), moving_avg, alpha=0.3, color='darkred')
            else:
                axes[1, 1].plot(agent.rl_agent.losses, color='darkred', linewidth=1, alpha=0.7)
            
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('RL Agent Loss Evolution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'agent_{agent_idx + 1}_{agents_short[agent_idx].replace("/", "_")}_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.show()


def plot_ml_vs_hybrid_detailed(all_agents, splits, ml_results, hybrid_results):
    """Create detailed confusion matrices and classification reports for each agent."""
    
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
    print("="*80)
    
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
    agent_names_list = list(ml_results.keys())
    test_datasets = [
        (splits[1][2], splits[1][5]),
        (splits[2][2], splits[2][5]),
        (splits[3][2], splits[3][5]),
        (splits[4][2], splits[4][5]),
        (splits[5][2], splits[5][5])
    ]
    
    for agent_idx, (agent, (X_test, y_test)) in enumerate(zip(all_agents, test_datasets)):
        print(f"\n📊 Confusion matrices for {agent_names_list[agent_idx]}...")
        
        # ML predictions
        y_pred_ml = agent.detector.predict(X_test, verbose=0)
        y_pred_ml = np.argmax(y_pred_ml, axis=1)
        
        # Hybrid predictions
        y_pred_hybrid = []
        for sample_idx in range(len(X_test)):
            if len(X_test[sample_idx].shape) == 3:
                sample = X_test[sample_idx].reshape(1, 28, 28, 1)
            else:
                sample = X_test[sample_idx].reshape(1, -1)
            
            pred = agent.detector.predict(sample, verbose=0)[0]
            confidence = np.max(pred)
            pred_class = np.argmax(pred)
            
            state = np.array([
                confidence, pred_class / agent.rl_agent.action_size,
                1 if pred_class == y_test[sample_idx] else 0,
                np.mean(pred), np.std(pred), np.min(pred), np.max(pred),
                len(pred), y_test[sample_idx] / agent.rl_agent.action_size,
                sample_idx / len(X_test)
            ])
            
            rl_action = agent.rl_agent.act(state, explore=False)
            y_pred_hybrid.append(rl_action)
        
        y_pred_hybrid = np.array(y_pred_hybrid)
        
        # Create confusion matrices
        cm_ml = confusion_matrix(y_test, y_pred_ml)
        cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{agent_names_list[agent_idx]} - Confusion Matrices', fontsize=14, fontweight='bold')
        
        # ML confusion matrix
        im1 = axes[0].imshow(cm_ml, cmap='Blues', aspect='auto')
        axes[0].set_title('ML Detector Only')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        
        # Add text annotations
        for i in range(cm_ml.shape[0]):
            for j in range(cm_ml.shape[1]):
                text = axes[0].text(j, i, cm_ml[i, j], ha="center", va="center", 
                                   color="white" if cm_ml[i, j] > cm_ml.max() / 2 else "black")
        
        plt.colorbar(im1, ax=axes[0])
        
        # Hybrid confusion matrix
        im2 = axes[1].imshow(cm_hybrid, cmap='Oranges', aspect='auto')
        axes[1].set_title('Hybrid ML+RL')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        
        # Add text annotations
        for i in range(cm_hybrid.shape[0]):
            for j in range(cm_hybrid.shape[1]):
                text = axes[1].text(j, i, cm_hybrid[i, j], ha="center", va="center",
                                   color="white" if cm_hybrid[i, j] > cm_hybrid.max() / 2 else "black")
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        filename = f'confusion_matrix_agent_{agent_idx + 1}_{agents_short[agent_idx]}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.show()


def plot_agent_comparison_grid(ml_results, hybrid_results):
    """Create a comprehensive grid comparing all agents."""
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE COMPARISON GRID")
    print("="*80)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('All Agents - ML vs Hybrid Performance Comparison', fontsize=16, fontweight='bold')
    
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
    agent_names_list = list(ml_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    for agent_idx, agent_name in enumerate(agent_names_list):
        ml_m = ml_results[agent_name]
        hybrid_m = hybrid_results[agent_name]
        
        # Top row: Metrics
        ml_vals = [ml_m[k] for k in metric_keys]
        hybrid_vals = [hybrid_m[k] for k in metric_keys]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, agent_idx].bar(x - width/2, ml_vals, width, label='ML', color='steelblue', alpha=0.8)
        axes[0, agent_idx].bar(x + width/2, hybrid_vals, width, label='Hybrid', color='coral', alpha=0.8)
        axes[0, agent_idx].set_title(agents_short[agent_idx], fontweight='bold', fontsize=12)
        axes[0, agent_idx].set_ylim([0, 1.0])
        axes[0, agent_idx].grid(True, alpha=0.3, axis='y')
        if agent_idx == 0:
            axes[0, agent_idx].legend(fontsize=9)
        
        # Bottom row: Improvement
        improvements = [hybrid_vals[i] - ml_vals[i] for i in range(len(metrics))]
        colors_imp = ['green' if x >= 0 else 'red' for x in improvements]
        
        axes[1, agent_idx].bar(x, improvements, color=colors_imp, alpha=0.7, width=0.6)
        axes[1, agent_idx].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, agent_idx].set_xticks(x)
        axes[1, agent_idx].set_xticklabels([m[:3] for m in metrics], fontsize=8, rotation=0)
        axes[1, agent_idx].set_ylim([-0.1, 0.1])
        axes[1, agent_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('all_agents_comparison_grid.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: all_agents_comparison_grid.png")
    plt.show()


def plot_learning_curves(learning_curves, n_episodes):
    """Plot learning curves for all agents during RL training."""
    
    print("\n[VISUALIZATION] Generating learning curves...")
    
    agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
    colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'purple']
    
    # Plot 1: Rewards over episodes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RL Training Learning Curves - Rewards', fontsize=16, fontweight='bold')
    
    for agent_idx in range(5):
        row = agent_idx // 3
        col = agent_idx % 3
        
        episodes = range(1, n_episodes + 1)
        rewards = learning_curves[agent_idx]['rewards']
        avg_rewards = learning_curves[agent_idx]['avg_rewards']
        
        axes[row, col].plot(episodes, rewards, alpha=0.3, color=colors[agent_idx], linewidth=1, label='Raw Rewards')
        axes[row, col].plot(episodes, avg_rewards, color=colors[agent_idx], linewidth=2, label='Moving Avg (window=10)')
        axes[row, col].set_title(f'{agents_short[agent_idx]} - Rewards', fontweight='bold', fontsize=12)
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Total Reward')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend(fontsize=9)
        
        # Add final reward annotation
        final_reward = avg_rewards[-1]
        axes[row, col].text(0.98, 0.02, f'Final: {final_reward:.2f}', 
                           transform=axes[row, col].transAxes,
                           ha='right', va='bottom', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove last empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('learning_curves_rewards.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: learning_curves_rewards.png")
    plt.show()
    
    # Plot 2: Losses over episodes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RL Training Learning Curves - Losses', fontsize=16, fontweight='bold')
    
    for agent_idx in range(5):
        row = agent_idx // 3
        col = agent_idx % 3
        
        episodes = range(1, n_episodes + 1)
        losses = learning_curves[agent_idx]['losses']
        avg_losses = learning_curves[agent_idx]['avg_losses']
        
        axes[row, col].plot(episodes, losses, alpha=0.3, color=colors[agent_idx], linewidth=1, label='Raw Loss')
        axes[row, col].plot(episodes, avg_losses, color=colors[agent_idx], linewidth=2, label='Moving Avg (window=10)')
        axes[row, col].set_title(f'{agents_short[agent_idx]} - Loss', fontweight='bold', fontsize=12)
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Average Loss')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend(fontsize=9)
        
        # Add final loss annotation
        final_loss = avg_losses[-1]
        axes[row, col].text(0.98, 0.02, f'Final: {final_loss:.4f}', 
                           transform=axes[row, col].transAxes,
                           ha='right', va='bottom', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Remove last empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('learning_curves_losses.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: learning_curves_losses.png")
    plt.show()
    
    # Plot 3: Comparison of all agents together
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('RL Training - All Agents Comparison', fontsize=16, fontweight='bold')
    
    episodes = range(1, n_episodes + 1)
    
    # Rewards comparison
    for agent_idx in range(5):
        avg_rewards = learning_curves[agent_idx]['avg_rewards']
        ax1.plot(episodes, avg_rewards, label=agents_short[agent_idx], color=colors[agent_idx], linewidth=2)
    
    ax1.set_title('Moving Average Rewards Across All Agents', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward per Step')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Losses comparison
    for agent_idx in range(5):
        avg_losses = learning_curves[agent_idx]['avg_losses']
        ax2.plot(episodes, avg_losses, label=agents_short[agent_idx], color=colors[agent_idx], linewidth=2)
    
    ax2.set_title('Moving Average Loss Across All Agents', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Loss')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: learning_curves_comparison.png")
    plt.show()
    
    print("\n[VISUALIZATION] Learning curves generated successfully!")
