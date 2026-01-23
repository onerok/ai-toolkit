'use client';

import { Listbox, ListboxButton, ListboxOption, ListboxOptions } from '@headlessui/react';
import { LuArrowUpDown, LuLoader } from 'react-icons/lu';

export type SortMode = 'filename' | 'date' | 'similarity';

const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'filename', label: 'Filename' },
  { value: 'date', label: 'Date Added' },
  { value: 'similarity', label: 'Similarity (pHash)' },
];

interface SortDropdownProps {
  value: SortMode;
  onChange: (mode: SortMode) => void;
  loading?: boolean;
}

export default function SortDropdown({ value, onChange, loading }: SortDropdownProps) {
  const selectedOption = SORT_OPTIONS.find(o => o.value === value)!;

  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative">
        <ListboxButton className="flex items-center gap-2 text-gray-200 bg-slate-600 px-3 py-1 rounded-md text-sm cursor-pointer">
          {loading ? <LuLoader className="animate-spin w-4 h-4" /> : <LuArrowUpDown className="w-4 h-4" />}
          <span>{selectedOption.label}</span>
        </ListboxButton>
        <ListboxOptions className="absolute right-0 mt-1 w-48 bg-slate-700 border border-slate-600 rounded-md shadow-lg z-50 py-1">
          {SORT_OPTIONS.map(option => (
            <ListboxOption
              key={option.value}
              value={option.value}
              className="cursor-pointer select-none px-3 py-1.5 text-sm text-gray-200 data-[focus]:bg-slate-600"
            >
              {option.label}
            </ListboxOption>
          ))}
        </ListboxOptions>
      </div>
    </Listbox>
  );
}
