import type { ThemeConfig } from 'antd'

/**
 * Ant Design theme mapped onto our monochrome light-glass system, so antd
 * controls (Select, DatePicker, TimePicker) match the rest of the UI: Manrope
 * type, near-black `#14141a` accent, soft hairline borders, 12px radius and a
 * glassy elevated popup. Applied globally via `<ConfigProvider>` in main.tsx.
 */
export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: '#14141a',
    colorInfo: '#14141a',
    colorText: '#14141a',
    colorTextSecondary: '#52525b',
    colorTextTertiary: '#71717a',
    colorTextPlaceholder: '#a1a1aa',
    colorBorder: 'rgba(17, 17, 26, 0.12)',
    colorBorderSecondary: 'rgba(17, 17, 26, 0.08)',
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',
    colorPrimaryHover: '#2a2a33',
    borderRadius: 12,
    borderRadiusLG: 14,
    borderRadiusSM: 8,
    controlHeight: 40,
    controlOutline: 'rgba(17, 17, 26, 0.12)',
    controlOutlineWidth: 2,
    fontFamily: '"Manrope", ui-sans-serif, system-ui, sans-serif',
    fontSize: 14,
    boxShadowSecondary:
      '0 12px 40px -12px rgba(17, 17, 26, 0.22), 0 0 0 1px rgba(17, 17, 26, 0.05)',
  },
  components: {
    Select: {
      optionSelectedBg: 'rgba(17, 17, 26, 0.06)',
      optionSelectedColor: '#14141a',
      optionSelectedFontWeight: 600,
      optionActiveBg: 'rgba(17, 17, 26, 0.04)',
      borderRadiusLG: 12,
    },
    DatePicker: {
      cellActiveWithRangeBg: 'rgba(17, 17, 26, 0.06)',
      cellHoverBg: 'rgba(17, 17, 26, 0.04)',
    },
    Input: { paddingBlock: 8 },
  },
}
