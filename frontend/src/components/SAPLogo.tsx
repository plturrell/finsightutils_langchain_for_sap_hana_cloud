import React from 'react';

interface SAPLogoProps {
  height?: number;
  color?: string;
}

const SAPLogo: React.FC<SAPLogoProps> = ({ height = 24, color = '#0066B3' }) => {
  return (
    <svg
      width={height * 1.2}
      height={height}
      viewBox="0 0 60 50"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M59.4,21.3H43.5c-1.3,0-2.7,1.1-3.1,2.3L35.1,43c-0.4,1.3,0.4,2.3,1.7,2.3h15.9c1.3,0,2.7-1.1,3.1-2.3l5.3-19.4
        C61.5,22.3,60.7,21.3,59.4,21.3z M51.7,38.9h-7.6l3.3-11.3h7.6L51.7,38.9z"
        fill={color}
      />
      <path
        d="M25.2,21.3l-5.3,19.4c-0.4,1.3,0.4,2.3,1.7,2.3h8c1.3,0,2.7-1.1,3.1-2.3l2.2-8.1c0.4-1.3-0.4-2.3-1.7-2.3h-7.1
        l1.9-6.7h10.9c1.3,0,2.7-1.1,3.1-2.3l0.5-1.9c0.4-1.3-0.4-2.3-1.7-2.3H25.2z"
        fill={color}
      />
      <path
        d="M0.5,21.3l-0.5,1.9c-0.4,1.3,0.4,2.3,1.7,2.3h8.9l-4.4,16.1c-0.4,1.3,0.4,2.3,1.7,2.3h7.6c1.3,0,2.7-1.1,3.1-2.3
        l4.4-16.1h8.9c1.3,0,2.7-1.1,3.1-2.3l0.5-1.9c0.4-1.3-0.4-2.3-1.7-2.3H3.7C2.4,19,1,20.1,0.5,21.3z"
        fill={color}
      />
    </svg>
  );
};

export default SAPLogo;