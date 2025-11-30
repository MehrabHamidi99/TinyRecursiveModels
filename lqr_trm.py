"""
LQR-inspired optimizer for TinyRecursiveModels

This is a simplified PyTorch implementation inspired by the JAX LQR optimizer.
It implements a diagonal preconditioner updated via layer-wise Jacobian approximations.

CURRENT STATUS:
- Preconditioner updates are DISABLED for TRM models due to their special forward() signature
- The optimizer currently works like Adam with identity preconditioner
- Infrastructure is in place for future improvements
- This is still useful as a baseline and for testing the integration

For a full LQR implementation, you would need to port:
- lqr_optimizer/_src/preconditioner.py
- lqr_optimizer/_src/utils/build_lqr.py  
- lqr_optimizer/_src/block_matrices_approx/block_structures.py

Key differences from full LQR:
1. Simplified to diagonal preconditioner (scalable, easy to implement)
2. Uses PyTorch autograd instead of JAX's jvp/vjp
3. No full LQR problem solving (just gradient-based preconditioner updates)
4. Preconditioner updates disabled for TRM (currently behaves like Adam)
"""

from typing import Optional, List, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass
import copy


@dataclass
class LQRConfig:
    """Configuration for LQR-style optimizer"""
    # Preconditioner settings
    precond_lr: float = 1e-2
    precond_update_every: int = 100
    precond_update_steps: int = 10
    precond_batch_size: int = 128
    
    # Structure type
    block_structure: str = "diagonal"  # Options: "diagonal", "scalar", "dense"
    
    # EMA for preconditioner
    ema_decay: float = 0.9
    
    # Regularization
    damping: float = 1e-4
    normalize_grad: bool = True
    clip_precond_min: float = 1e-8  # Prevent gradient inversion
    
    # Main optimizer settings (for parameter updates)
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)


class DiagonalPreconditioner:
    """
    Diagonal preconditioner for gradients.
    
    This maintains a diagonal matrix P for each parameter tensor,
    such that preconditioned_grad = P @ grad.
    """
    
    def __init__(self, model: nn.Module, config: LQRConfig):
        self.config = config
        self.model = model
        
        # Initialize preconditioner matrices as ones (identity initially)
        self.precond_dict: Dict[str, torch.Tensor] = {}
        self.ema_precond_dict: Dict[str, torch.Tensor] = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Initialize as identity (ones for diagonal)
                self.precond_dict[name] = torch.ones_like(param.data)
                self.ema_precond_dict[name] = torch.ones_like(param.data)
    
    def apply(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply preconditioner to gradients"""
        precond_grads = {}
        for name, grad in gradients.items():
            if name in self.precond_dict:
                precond_grads[name] = self.precond_dict[name] * grad
            else:
                precond_grads[name] = grad
        return precond_grads
    
    def update(self, model: nn.Module, data_batch: torch.Tensor, 
               labels: torch.Tensor, loss_fn: Callable,
               main_optimizer: Optimizer, carry=None):
        """
        Update preconditioner based on curvature information.
        
        This is a simplified version that approximates the LQR approach:
        1. Compute gradients
        2. Estimate per-parameter curvature via gradient magnitude statistics
        3. Update preconditioner to adapt to curvature
        """
        # Skip update for now - TRM models have special forward signature
        # This makes LQR optimizer work like Adam for now
        # TODO: Implement proper preconditioner update for TRM models
        return
        
        # Original code below (disabled for TRM compatibility)
        with torch.no_grad():
            # Store current gradients for multiple steps
            grad_history = []
            
            for _ in range(self.config.precond_update_steps):
                # Forward pass
                model.zero_grad()
                
                # Try to handle different model interfaces
                try:
                    if carry is not None:
                        # TRM-style model with carry
                        outputs = model(carry=carry, batch=data_batch)
                        if isinstance(outputs, tuple):
                            carry, outputs = outputs[0], outputs[1]
                    else:
                        # Standard model
                        outputs = model(data_batch)
                except TypeError:
                    # If model signature doesn't match, skip this update
                    print("Warning: Could not update preconditioner - model interface mismatch")
                    return
                
                loss = loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Collect gradients
                current_grads = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        current_grads[name] = param.grad.clone()
                
                grad_history.append(current_grads)
            
            # Update preconditioner based on gradient statistics
            for name in self.precond_dict.keys():
                # Compute gradient variance/magnitude as curvature proxy
                grads_for_param = [gh[name] for gh in grad_history if name in gh]
                
                if len(grads_for_param) > 0:
                    # Stack gradients
                    grad_stack = torch.stack(grads_for_param)
                    
                    # Compute second moment (proxy for curvature)
                    grad_sq_mean = (grad_stack ** 2).mean(dim=0)
                    
                    # Update rule: preconditioner ∝ 1 / sqrt(curvature)
                    # This is similar to Adam/RMSprop but motivated by Newton's method
                    new_precond = 1.0 / (torch.sqrt(grad_sq_mean) + self.config.damping)
                    
                    # Normalize preconditioner
                    if self.config.normalize_grad:
                        new_precond = new_precond / (new_precond.mean() + 1e-8)
                    
                    # Clip to prevent gradient inversion
                    new_precond = torch.clamp(new_precond, min=self.config.clip_precond_min)
                    
                    # Apply EMA
                    if self.config.ema_decay > 0:
                        self.precond_dict[name] = (
                            self.config.ema_decay * self.ema_precond_dict[name] +
                            (1 - self.config.ema_decay) * new_precond
                        )
                        self.ema_precond_dict[name] = self.precond_dict[name].clone()
                    else:
                        self.precond_dict[name] = new_precond
    
    def state_dict(self) -> Dict[str, Any]:
        """Save preconditioner state"""
        return {
            'precond_dict': {k: v.cpu() for k, v in self.precond_dict.items()},
            'ema_precond_dict': {k: v.cpu() for k, v in self.ema_precond_dict.items()},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load preconditioner state"""
        self.precond_dict = {k: v.to(next(iter(self.model.parameters())).device) 
                            for k, v in state_dict['precond_dict'].items()}
        self.ema_precond_dict = {k: v.to(next(iter(self.model.parameters())).device)
                                for k, v in state_dict['ema_precond_dict'].items()}


class LQRAdamOptimizer(Optimizer):
    """
    Combined LQR preconditioner + Adam optimizer.
    
    This wraps a standard Adam optimizer with LQR-inspired preconditioning.
    """
    
    def __init__(self, params, model: nn.Module, config: LQRConfig):
        self.config = config
        self.model = model
        
        # Create preconditioner
        self.preconditioner = DiagonalPreconditioner(model, config)
        
        # Base optimizer (Adam)
        defaults = dict(
            lr=config.lr,
            betas=config.betas,
            eps=1e-8,
            weight_decay=config.weight_decay
        )
        super().__init__(params, defaults)
        
        # Track steps for preconditioner updates
        self.step_count = 0
    
    @torch.no_grad()
    def step(self, closure=None, data_batch=None, labels=None, loss_fn=None, carry=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model
            data_batch: Batch data for preconditioner update (not used for TRM)
            labels: Labels for preconditioner update (not used for TRM)
            loss_fn: Loss function for preconditioner update (not used for TRM)
            carry: Carry state for TRM models (not used for TRM)
        
        Note: Preconditioner updates are currently disabled for TRM models
        due to special forward() signature. The optimizer works like Adam.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update preconditioner periodically
        # Currently disabled for TRM - see update() method
        if (self.step_count > 0 and 
            self.step_count % self.config.precond_update_every == 0 and
            data_batch is not None and labels is not None and loss_fn is not None):
            if self.step_count % (self.config.precond_update_every * 10) == 0:
                # Only print occasionally to avoid spam
                print(f"[LQR Note] Preconditioner updates disabled for TRM models (step {self.step_count})")
            # Skip update for TRM compatibility
            # self.preconditioner.update(self.model, data_batch, labels, loss_fn, self, carry)
        
        # Collect gradients
        gradients = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Get parameter name (this is tricky without name mapping)
                    param_name = None
                    for name, param in self.model.named_parameters():
                        if param is p:
                            param_name = name
                            break
                    
                    if param_name:
                        gradients[param_name] = p.grad.clone()
        
        # Apply preconditioner
        precond_grads = self.preconditioner.apply(gradients)
        
        # Apply preconditioned gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Find parameter name
                param_name = None
                for name, param in self.model.named_parameters():
                    if param is p:
                        param_name = name
                        break
                
                if param_name and param_name in precond_grads:
                    grad = precond_grads[param_name]
                else:
                    grad = p.grad
                
                # Adam update
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        self.step_count += 1
        return loss


class LQRSGDOptimizer(Optimizer):
    """
    Combined LQR preconditioner + SGD optimizer.
    
    This wraps a standard SGD optimizer with LQR-inspired preconditioning.
    """
    
    def __init__(self, params, model: nn.Module, config: LQRConfig, momentum: float = 0.9):
        self.config = config
        self.model = model
        self.momentum = momentum
        
        # Create preconditioner
        self.preconditioner = DiagonalPreconditioner(model, config)
        
        # Base optimizer (SGD)
        defaults = dict(
            lr=config.lr,
            momentum=momentum,
            weight_decay=config.weight_decay
        )
        super().__init__(params, defaults)
        
        # Track steps for preconditioner updates
        self.step_count = 0
    
    @torch.no_grad()
    def step(self, closure=None, data_batch=None, labels=None, loss_fn=None, carry=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model
            data_batch: Batch data for preconditioner update (not used for TRM)
            labels: Labels for preconditioner update (not used for TRM)
            loss_fn: Loss function for preconditioner update (not used for TRM)
            carry: Carry state for TRM models (not used for TRM)
        
        Note: Preconditioner updates are currently disabled for TRM models
        due to special forward() signature. The optimizer works like SGD.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update preconditioner periodically
        # Currently disabled for TRM - see update() method
        if (self.step_count > 0 and 
            self.step_count % self.config.precond_update_every == 0 and
            data_batch is not None and labels is not None and loss_fn is not None):
            if self.step_count % (self.config.precond_update_every * 10) == 0:
                # Only print occasionally to avoid spam
                print(f"[LQR Note] Preconditioner updates disabled for TRM models (step {self.step_count})")
            # Skip update for TRM compatibility
            # self.preconditioner.update(self.model, data_batch, labels, loss_fn, self, carry)
        
        # Collect gradients
        gradients = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Get parameter name (this is tricky without name mapping)
                    param_name = None
                    for name, param in self.model.named_parameters():
                        if param is p:
                            param_name = name
                            break
                    
                    if param_name:
                        gradients[param_name] = p.grad.clone()
        
        # Apply preconditioner
        precond_grads = self.preconditioner.apply(gradients)
        
        # Apply preconditioned gradients with SGD update
        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Find parameter name
                param_name = None
                for name, param in self.model.named_parameters():
                    if param is p:
                        param_name = name
                        break
                
                if param_name and param_name in precond_grads:
                    grad = precond_grads[param_name]
                else:
                    grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # SGD with momentum
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-group['lr'])
        
        self.step_count += 1
        return loss


# Example usage function
def create_lqr_adam_optimizer_for_trm(model: nn.Module, config: LQRConfig) -> LQRAdamOptimizer:
    """
    Create LQR-inspired optimizer with Adam for TinyRecursiveModels.
    
    Usage in pretrain.py:
        from lqr_trm import create_lqr_adam_optimizer_for_trm, LQRConfig
        
        lqr_config = LQRConfig(
            lr=1e-4,
            precond_lr=1e-2,
            precond_update_every=100,
            ema_decay=0.9
        )
        optimizer = create_lqr_adam_optimizer_for_trm(model, lqr_config)
        
        # In training loop:
        loss.backward()
        optimizer.step(data_batch=x, labels=y, loss_fn=criterion)
    """
    return LQRAdamOptimizer(model.parameters(), model, config)


def create_lqr_sgd_optimizer_for_trm(model: nn.Module, config: LQRConfig, momentum: float = 0.9) -> LQRSGDOptimizer:
    """
    Create LQR-inspired optimizer with SGD for TinyRecursiveModels.
    
    Usage in pretrain.py:
        from lqr_trm import create_lqr_sgd_optimizer_for_trm, LQRConfig
        
        lqr_config = LQRConfig(
            lr=1e-4,
            precond_lr=1e-2,
            precond_update_every=100,
            ema_decay=0.9
        )
        optimizer = create_lqr_sgd_optimizer_for_trm(model, lqr_config, momentum=0.9)
        
        # In training loop:
        loss.backward()
        optimizer.step(data_batch=x, labels=y, loss_fn=criterion)
    """
    return LQRSGDOptimizer(model.parameters(), model, config, momentum=momentum)


# Keep backward compatibility
def create_lqr_optimizer_for_trm(model: nn.Module, config: LQRConfig) -> LQRAdamOptimizer:
    """
    Create LQR-inspired optimizer for TinyRecursiveModels.
    
    Usage in pretrain.py:
        from lqr_trm import create_lqr_optimizer_for_trm, LQRConfig
        
        lqr_config = LQRConfig(
            lr=1e-4,
            precond_lr=1e-2,
            precond_update_every=100,
            ema_decay=0.9
        )
        optimizer = create_lqr_optimizer_for_trm(model, lqr_config)
        
        # In training loop:
        loss.backward()
        optimizer.step(data_batch=x, labels=y, loss_fn=criterion)
    """
    return LQRAdamOptimizer(model.parameters(), model, config)


# TODO: For full LQR implementation, you would need to port:
# 1. Full LQR problem formulation (build_lqr.py)
# 2. Block matrix structures (KFAC, dense, etc.)
# 3. LQR backward/forward pass with proper Riccati equation solving
# 4. Divergence functions (KL, Renyi, etc.)
#
# This would require ~1000+ lines of careful PyTorch translation from JAX.
# The current implementation provides a practical starting point with diagonal preconditioning.

