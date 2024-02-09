import React from 'react';
import { RadioGroup, Radio, Stack } from '@chakra-ui/react'

export default function RadioExample() {
    const [value, setValue] = React.useState('1')
    return (
        <RadioGroup onChange={setValue} value={value}>
            <Stack direction='row'>
                <Radio value='1'>First</Radio>
                <Radio value='2'>Second</Radio>
                <Radio value='3'>Third</Radio>
            </Stack>
        </RadioGroup>
    )
}
