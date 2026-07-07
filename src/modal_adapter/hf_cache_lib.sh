# Shared helpers for the Modal adapter phase scripts.
#
# hf_symlink_farm replaces run_task.sh's with_huggingface_overlay:
# fuse-overlayfs cannot mount inside Modal's gVisor runtime, so instead of an
# overlay mount we build a "symlink farm". Every directory of the read-only
# HF cache volume is recreated as a real (writable) directory on container
# disk, and every file becomes a symlink into the read-only volume. Existing
# cache content is read through the symlinks; new downloads land on container
# disk and are discarded with the container -- the same throwaway-upper-layer
# semantics as the overlay. The one difference: overwriting an already-cached
# file in place fails (read-only) instead of copying up.

hf_symlink_farm() {
    local lower="$1"
    local target="$2"
    mkdir -p "$target"
    if [ -d "$lower" ] && [ -n "$(ls -A "$lower" 2>/dev/null)" ]; then
        cp -rs "$lower/." "$target/"
    fi
}

# Print `-u VAR` arguments for env(1) that strip Modal-internal variables from
# the environment handed to agent/judge subprocesses. On the cluster the
# containers inherit the host environment; the Modal equivalent should not
# leak MODAL_* runtime variables (e.g. the identity token) to the agent.
modal_env_unsets() {
    compgen -e | grep '^MODAL_' | sed 's/^/-u /' | tr '\n' ' '
    true
}
