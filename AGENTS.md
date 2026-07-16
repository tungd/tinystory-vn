# Agent instructions

- Be extremely concise. Sacrifice grammar for concision.
- Use built-in plan mode, not a writing-plans skill.

## Colab Drive authorization

`colab drivemount` authorization is tied to its live waiter and one-time state.
Do not run it in a tool session that will close while waiting for the user; a
rerun creates a different link and invalidates the completed flow.

Keep the waiter in persistent tmux:

```bash
tmux new-session -d -s SESSION-drive-auth 'colab drivemount -s SESSION'
tmux capture-pane -p -J -t SESSION-drive-auth -S -60
```

Give the user the exact unwrapped URL from `capture-pane -J`. After they say
`done`, resume the same waiter:

```bash
tmux send-keys -t SESSION-drive-auth Enter
tmux capture-pane -p -J -t SESSION-drive-auth -S -60
```

Wait for mount success before detaching/letting tmux exit. Never rerun
`drivemount` after the user authorizes the existing link.
