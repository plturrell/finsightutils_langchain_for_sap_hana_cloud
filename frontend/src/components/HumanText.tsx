import React from 'react';
import { Typography, TypographyProps } from '@mui/material';
import { humanize } from '../utils/humanLanguage';

interface HumanTextProps extends Omit<TypographyProps, 'children'> {
  children: string;
}

/**
 * HumanText component automatically converts technical terminology
 * to human-friendly language in all text content.
 * 
 * Usage:
 * <HumanText variant="body1">Configure Vector Search with TensorRT</HumanText>
 * Renders as: "Set up Meaning Search with Performance Engine"
 */
const HumanText: React.FC<HumanTextProps> = ({ children, ...props }) => {
  // Only process string children
  const humanizedText = typeof children === 'string' 
    ? humanize(children) 
    : children;
  
  return (
    <Typography {...props}>
      {humanizedText}
    </Typography>
  );
};

export default HumanText;