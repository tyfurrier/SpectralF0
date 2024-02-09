import { useState, useEffect } from 'react';

// icons
import {
    IoPlayBackSharp,
    IoPlayForwardSharp,
    IoPlaySkipBackSharp,
    IoPlaySkipForwardSharp,
    IoPlaySharp,
    IoPauseSharp,
} from 'react-icons/io5';

const Controls = ({ playing, setPlaying, audioRefs, trackNumber }) => {
    let isPlaying = false;

    const togglePlayPause = () => {
        console.log(`audio refs: ${audioRefs}`)
        if (playing != trackNumber) {
            isPlaying = true;
            setPlaying(trackNumber);
        } else {
            isPlaying = false;
            setPlaying(-1);
        }
    };
    // ...
    useEffect(() => {
        console.log(`audiorefs: ${audioRefs.current}`)
        console.log(`playing: ${playing}`)
        if (playing == trackNumber) {
            audioRefs.current[trackNumber].play();
        } else {
            audioRefs.current[trackNumber].pause();
        }
    }, [playing]);

    return (
        <div className="controls-wrapper">
            <div className="controls">
                <button>
                    <IoPlaySkipBackSharp />
                </button>
                <button>
                    <IoPlayBackSharp />
                </button>

                <button onClick={togglePlayPause}>
                    {(playing == trackNumber) ? <IoPauseSharp /> : <IoPlaySharp />}
                </button>
                <button>
                    <IoPlayForwardSharp />
                </button>
                <button>
                    <IoPlaySkipForwardSharp />
                </button>
            </div>
        </div>
    );
};

export default Controls;