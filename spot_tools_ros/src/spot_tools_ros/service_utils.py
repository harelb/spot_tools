import threading


def wait_for_future(future, timeout_s):
    """Block until an rclpy future is done, or timeout_s passes.

    Waits on an Event set by a done-callback instead of polling, so the caller
    sleeps while the executor delivers the reply. The client must be in a
    callback group the executor can service while the caller waits (i.e. not a
    mutually exclusive group the caller itself is running in).

    Returns True if the future finished in time.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if future.done():
        done.set()
    return done.wait(timeout_s)


def call_service_with_retry(
    client,
    request,
    logger,
    description,
    attempts=3,
    wait_for_service_s=3.0,
    call_timeout_s=5.0,
):
    """Call a ROS service, retrying when it is unavailable or does not answer.

    Meant for calls from a worker thread (not an executor callback). Returns
    the response, or None after the last attempt failed. Every failed attempt
    is logged at WARNING and the final failure at ERROR, with `description`
    naming what the call was for.
    """
    attempts = max(1, int(attempts))
    reason = "not attempted"
    for attempt in range(1, attempts + 1):
        tag = f"{description} (attempt {attempt}/{attempts})"
        if not client.wait_for_service(timeout_sec=wait_for_service_s):
            reason = f"service not available after {wait_for_service_s:.1f} s"
            logger.warning(f"{tag}: {reason}")
            continue

        future = client.call_async(request)
        if not wait_for_future(future, call_timeout_s):
            reason = f"no reply within {call_timeout_s:.1f} s"
            logger.warning(f"{tag}: {reason}")
            remove = getattr(client, "remove_pending_request", None)
            if remove is not None:
                remove(future)
            continue

        exc = future.exception()
        if exc is not None:
            reason = f"call raised {exc!r}"
            logger.warning(f"{tag}: {reason}")
            continue

        response = future.result()
        if response is None:
            reason = "empty response"
            logger.warning(f"{tag}: {reason}")
            continue
        return response

    logger.error(f"{description} FAILED after {attempts} attempt(s): {reason}")
    return None
