class EventScheduler:
    """
    Handles note on and off events
    Can handle up to the specified number of simultaneous voices
    You provide a function that produces a synth (no ADSR): `synth_factory`
    You provide a function that produces an ADSR instance, if desired: `adsr_factory`
    The scheduler will create and manage the synths and ADSRs as needed
    Under the assumption that most tracks have some significant ADSR envelope
    The voice-stealing rule is simple:
        If utilizing ADSR:
            If there are any notes in the release phase:
                Steal the oldest one (earliest note off time)
            if not:
                Steal the oldest note on (earliest note on time)
        else:
            note_off events immediately free the voice
            So a steal will occur only to note_on when all voices are active
            It will simply steal the oldest note on
    
    Algorithm for achieveing time accuracy:
        Many synths do not have a well-defined seek() function
        Particularly, if the synth is very stateful

        So... suppose that we have a note_on event
        that occurs X samples after the start of the current processing block
        We simply start a voice (reset it), and then silently generate X samples
        and discard them (a.k.a. "burn X samples"), then, we generate STEP_SIZE - X samples
        and mix at the correct location into the block buffer 

        Note:
            Make sure note reset and ADSR (if present) reset are synchronized
        Note:
            Mid-buffer note-offs are easier to handle as we simply generate more
            samples from the ongoing stateful voice


    """

    # todo