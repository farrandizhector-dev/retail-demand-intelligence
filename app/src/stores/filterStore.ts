import { create } from "zustand";

interface FilterStore {
  selectedState: string;
  selectedCategory: string;
  selectedDepartment: string;
  setSelectedState: (v: string) => void;
  setSelectedCategory: (v: string) => void;
  setSelectedDepartment: (v: string) => void;
}

export const useFilterStore = create<FilterStore>((set) => ({
  selectedState: "all",
  selectedCategory: "all",
  selectedDepartment: "all",
  setSelectedState: (v) => set({ selectedState: v }),
  setSelectedCategory: (v) => set({ selectedCategory: v }),
  setSelectedDepartment: (v) => set({ selectedDepartment: v }),
}));

export const STATES = ["CA", "TX", "WI"];
export const CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"];
export const DEPARTMENTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"];
