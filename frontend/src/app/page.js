'use client'

import Image from 'next/image'
import theme from '../theme'
import React, { useState } from 'react';
import { ChakraProvider, Radio, RadioGroup, Stack } from '@chakra-ui/react'
import getCombo from '../data/sounds'
import AudioPlayer from '@/components/AudioPlayer';

// import components
import DisplayTrack from '../components/DisplayAudio';
import Controls from '../components/Controls';
import RadioExample from '@/components/Radio';

export default function Home() {
  const [playing, setPlaying] = useState(-1);
  const [decision, setDecision] = useState(2);
  console.log(`playing: ${playing}`)
  return (
    <ChakraProvider theme={theme}>
      <Stack>
        <AudioPlayer {...{ playing, setPlaying }}>
        </AudioPlayer>
        <RadioExample />
      </Stack>
    </ChakraProvider>
  )
}
