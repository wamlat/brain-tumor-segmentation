import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast # Keep AMP components

from models.dataset import BraTSDataset
from models.unet import UNet

if __name__ == '__main__':
    # Settings
    BATCH_SIZE = 8
    NUM_EPOCHS = 150
    LEARNING_RATE = 2e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = False # <--- *** KEEPING AMP ENABLED ***

    print(f"Using device: {DEVICE}")
    print(f"Automatic Mixed Precision (AMP): {USE_AMP}")

    os.makedirs('./outputs/checkpoints', exist_ok=True)
    os.makedirs('./outputs/loss_plots', exist_ok=True)

    train_dataset = BraTSDataset(root_dir='./data_original/raw/GLI-Training')
    print(f"Total available patients: {len(train_dataset.subjects)}")
    # Consider reducing num_workers if you encounter DataLoader issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet(in_channels=4, out_channels=1)
    model = model.to(DEVICE)

    print("Training with all parameters unfrozen from the start")

    pos_weight = torch.tensor([2.5], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # Initialize GradScaler only if using CUDA and AMP
    scaler = GradScaler(enabled=USE_AMP and DEVICE.type == 'cuda')

    checkpoint_path = './outputs/checkpoints/best_model.pth'
    start_epoch = 0
    best_loss = float('inf')
    loss_history = []  # Store history for plotting

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)  # Consider weights_only=True
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            checkpoint = None  # Ensure we start fresh if loading fails badly

        if checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error loading model state_dict: {e}. Check model definition. Starting fresh.")
                checkpoint = None  # Treat as fresh start if model doesn't match

        if checkpoint:  # Proceed only if checkpoint and model loaded successfully
            optimizer_loaded_successfully = False
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
                # Lower LR when resuming with existing optimizer state
                print("Lowering learning rate for continued training.")
                current_lr = 5e-5  # Define the LR to use
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                optimizer_loaded_successfully = True  # Mark as successful

            except (ValueError, KeyError) as e:
                print(f"Optimizer state mismatch or key error ({e}), creating fresh optimizer with lower LR.")
                # Create a new optimizer instance with the lower LR
                current_lr = 5e-5  # Define the LR to use
                optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-5)

                # *** FIX: Manually add initial_lr for the new optimizer ***
                print("Manually setting 'initial_lr' for new optimizer before scheduler init.")
                for group in optimizer.param_groups:
                    group['initial_lr'] = current_lr
                # *** End FIX ***

            start_epoch = checkpoint.get('epoch', -1) + 1  # Use .get for safety
            loss_history = checkpoint.get('loss_history', [])
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch}, loaded best loss: {best_loss:.4f}")

            # Initialize scheduler correctly AFTER optimizer is loaded/created and potentially modified
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=start_epoch - 1
            )
            print(f"Scheduler initialized with last_epoch={start_epoch - 1}")

        else:  # Handle cases where checkpoint loading failed earlier
            print("Checkpoint invalid or model mismatch, starting fresh training.")
            start_epoch = 0
            best_loss = float('inf')
            loss_history = []
            # Ensure optimizer and scheduler are created for fresh start
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            # No need to set last_epoch here, defaults to -1

    else:  # Handle case where checkpoint file doesn't exist
        print("No checkpoint found, starting fresh training.")
        start_epoch = 0
        best_loss = float('inf')
        loss_history = []
        # Ensure optimizer and scheduler are created for fresh start
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # No need to set last_epoch here, defaults to -1

    # --- Plotting Setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot existing history if loaded
    epochs_plotted = list(range(1, len(loss_history) + 1))
    line, = ax.plot(epochs_plotted, loss_history, 'b-', label='Loss per Epoch')  # Plot raw loss

    # Calculate existing moving average history correctly
    avg_window = 5
    moving_avg_loss = []
    if len(loss_history) > 0:
        temp_ma_values = []  # Use a temporary list to build the MA history
        for i in range(len(loss_history)):
            # Check if enough data points exist for a full window
            if i >= avg_window - 1:
                window = loss_history[i - avg_window + 1: i + 1]
                if window:  # Ensure window is not empty
                    temp_ma_values.append(sum(window) / len(window))
            else:
                # Calculate progressive MA for the beginning
                window = loss_history[:i + 1]
                if window:  # Ensure window is not empty
                    temp_ma_values.append(sum(window) / len(window))
        moving_avg_loss = temp_ma_values  # Assign the calculated MA history

    # *** FIX: Correct x-axis for initial MA plot ***
    # X-axis should match the number of calculated MA points
    epochs_avg_plotted = list(range(1, len(moving_avg_loss) + 1))
    avg_line, = ax.plot(epochs_avg_plotted, moving_avg_loss, 'r-', linewidth=2,
                        label=f'Moving Average ({avg_window})')  # Plot initial MA

    # Setup axes limits and labels
    ax.set_xlim(0, max(NUM_EPOCHS, start_epoch + 10))  # Dynamic xlim based on total epochs

    # Dynamically set Y limit based on loaded data, filtering potential NaNs
    all_valid_losses = [loss for loss in (loss_history + moving_avg_loss) if not torch.isnan(torch.tensor(loss))]
    if all_valid_losses:
        ax.set_ylim(0, max(0.5, max(all_valid_losses) * 1.2))
    else:
        ax.set_ylim(0, 0.5)  # Default if no valid losses yet

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend()  # Add legend
    plt.grid(True)
    # --- End Plotting Setup ---


    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        total_batches = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            try:
                images, masks = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch {batch_idx}: {e}. Skipping batch.")
                continue # Skip this batch if data is malformed

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{total_batches}", end="\r")

            images = images.to(DEVICE, non_blocking=True)
            masks = (masks > 0).float().to(DEVICE, non_blocking=True) # Ensure float 0/1 masks

            optimizer.zero_grad(set_to_none=True) # Zero grads before forward pass

            # --- Forward Pass with AMP ---
            with autocast(enabled=USE_AMP, device_type=DEVICE.type):
                outputs = model(images)
                # --- Check Outputs within Autocast ---
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n!!! NaN/Inf detected in MODEL OUTPUTS (inside autocast) at epoch {epoch+1}, batch {batch_idx}. Skipping batch. !!!")
                    outputs = torch.nan_to_num(outputs) # Attempt to replace NaN/Inf to prevent loss NaN, or just skip
                    # continue # Or skip batch entirely
                # --- End Check ---
                loss = criterion(outputs, masks)

            # --- Check Loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! Loss became NaN/Inf at epoch {epoch+1}, batch {batch_idx}. Skipping backprop. !!!")
                # Optional: print stats of outputs/masks that caused NaN loss
                # print(f"Output stats before loss: min={outputs.min().item()}, max={outputs.max().item()}")
                continue # Skip backprop and optimizer step

            # --- Backward Pass with Scaler ---
            # scaler.scale(loss).backward() should be robust to NaN/Inf loss, but checking before is safer
            scaler.scale(loss).backward()

            # --- Gradient Unscaling and Clipping with Checks ---
            # Unscale first
            scaler.unscale_(optimizer)

            # *** Check Gradients for NaN/Inf BEFORE clipping ***
            found_nan_inf_grad = False
            total_grad_norm_before_clip = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        # print(f"\n!!! NaN/Inf detected in GRADIENTS for param BEFORE clipping at epoch {epoch+1}, batch {batch_idx}. !!!")
                        found_nan_inf_grad = True
                        param.grad = torch.nan_to_num(param.grad) # Replace NaN/Inf in grad
                        # break # Stop checking once found
                    # Calculate norm contribution safely
                    # param_norm = torch.nan_to_num(param.grad.data).norm(2) # More robust norm calc
                    # total_grad_norm_before_clip += param_norm.item() ** 2
            # total_grad_norm_before_clip = total_grad_norm_before_clip ** 0.5
            # if found_nan_inf_grad:
            #    print(f"  (NaN/Inf found in grads before clip)") # Indicate issue found

            # Clip gradients
            # grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Apply clipping regardless

            # --- Optimizer Step ---
            # scaler.step() checks for inf/nan gradients internally scaled by the scaler.
            # If found, it skips the optimizer step and warns.
            scaler.step(optimizer)

            # --- Scaler Update ---
            scaler.update()
            # --- End Backward Pass ---

            if not (torch.isnan(loss) or torch.isinf(loss)): # Only accumulate valid loss
                running_loss += loss.item()
                batch_count += 1


        # --- Epoch Summary and Plotting ---

        if batch_count == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], No valid batches processed, skipping epoch summary.")
            continue # Avoid division by zero if all batches were skipped

        avg_loss = running_loss / batch_count # Use batch_count
        loss_history.append(avg_loss)

        # Calculate Moving Average
        current_moving_avg = float('nan') # Default to nan if not enough history
        if len(loss_history) >= avg_window:
            current_moving_avg = sum(loss_history[-avg_window:]) / avg_window
            moving_avg_loss.append(current_moving_avg)
        else:
             # Optionally calculate MA progressively
             current_moving_avg = sum(loss_history) / len(loss_history)
             moving_avg_loss.append(current_moving_avg) # Append early MA or just avg_loss

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Moving Avg: {current_moving_avg:.4f}, LR: {current_lr:.6f}")

        # --- Update Plot ---
        # Update raw loss plot data
        line.set_xdata(range(1, len(loss_history) + 1))
        line.set_ydata(loss_history)

        # *** FIX: Correctly set x-data for moving average plot update ***
        # The x-values should simply correspond to the number of MA points calculated so far.
        avg_line.set_xdata(range(1, len(moving_avg_loss) + 1))
        avg_line.set_ydata(moving_avg_loss)
        # *** End FIX ***

        # Update plot limits and redraw
        ax.relim()
        ax.autoscale_view()
        try:  # Add try-except around draw/flush for robustness
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception as e:
            print(f"\nWarning: Plot drawing error: {e}")  # Non-fatal error for plotting
        plt.pause(0.01)  # Slightly longer pause
        # --- End Update Plot ---

        scheduler.step() # Step the scheduler each epoch

        # Save based on moving average loss
        if current_moving_avg < best_loss and not torch.isnan(torch.tensor(current_moving_avg)):
            best_loss = current_moving_avg
            save_path = './outputs/checkpoints/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss, # Save the best MA loss
                'loss_history': loss_history # Save history
            }, save_path)
            print(f"New best model saved at epoch {epoch + 1} with moving average loss {best_loss:.4f}")

        # Early Stopping Logic (optional, based on moving average)
        patience = 15
        if epoch > 30 and len(moving_avg_loss) >= patience:
             min_last_patience = min(moving_avg_loss[-patience:])
             min_before_patience = min(moving_avg_loss[:-patience]) if len(moving_avg_loss) > patience else float('inf')
             if min_last_patience >= min_before_patience: # Use >= to handle plateaus
                 print(f"No improvement in moving average loss for {patience} epochs. Early stopping.")
                 break

    # --- Final Plot Saving (Same as Option 1) ---
    plt.ioff()
    final_fig, final_ax = plt.subplots(figsize=(12, 8))
    final_epochs = range(1, len(loss_history) + 1)
    final_ax.plot(final_epochs, loss_history, 'b-', linewidth=1, alpha=0.7, label='Loss per Epoch')
    final_ax.plot(range(avg_window if len(loss_history) >= avg_window else 1, len(loss_history) + 1), moving_avg_loss, 'r-', linewidth=2, label=f'Moving Average ({avg_window})')
    final_ax.grid(True)
    final_ax.set_title('Final Training Loss Over Epochs')
    final_ax.set_xlabel('Epoch')
    final_ax.set_ylabel('Loss')
    final_ax.legend()
    if best_loss != float('inf') and moving_avg_loss:
         try:
             best_ma_loss_val = min(filter(lambda x: not torch.isnan(torch.tensor(x)), moving_avg_loss))
             best_epoch_idx = moving_avg_loss.index(best_ma_loss_val)
             best_epoch_num = (avg_window if len(loss_history) >= avg_window else 1) + best_epoch_idx
             final_ax.annotate(f'Best MA Loss: {best_ma_loss_val:.4f} (Epoch ~{best_epoch_num})',
                              xy=(best_epoch_num, best_ma_loss_val), xytext=(best_epoch_num + 5, best_ma_loss_val + 0.05),
                              arrowprops=dict(facecolor='black', shrink=0.05))
         except (ValueError, IndexError): print("Could not determine best epoch for annotation.")
    plot_save_path = './outputs/loss_plots/final_loss_curve.png'
    final_fig.savefig(plot_save_path, dpi=300)
    print(f"Final loss plot saved to '{plot_save_path}'")
    plt.show()

    print(f"Training completed.")
    print(f"Best validation loss (moving average): {best_loss:.4f}")

    # Save final model state
    final_model_path = './outputs/checkpoints/final_model.pth'
    torch.save({
        'epoch': epoch, # Last completed epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss if batch_count > 0 else float('nan'),
        'loss_history': loss_history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")